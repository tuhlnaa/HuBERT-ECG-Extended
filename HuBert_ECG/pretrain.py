import copy
from dataclasses import dataclass
from pathlib import Path
import torch
import wandb

import numpy as np
import torch.nn as nn
import torch.optim as optim

from loguru import logger
from math import ceil
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HubertConfig
from transformers import get_linear_schedule_with_warmup
# from transformers.models.hubert.modeling_hubert import compute_mask_indices

from torch.nn import functional as F

# Import custom modules
from dataset import ECGDataset
from config import create_training_parser, init_seeds
from hubert_ecg import HuBERTECG as HuBERT, HuBERTECGConfig

EPS = 1E-09
MINIMAL_IMPROVEMENT = 1e-3
DROPOUT_ADJUSTMENT = 0.05
WEIGHT_DECAY_MULTIPLIER = 5.0
SELF_SUPERVISED_MODEL_CKPT_PATH = "output/checkpoints/self-supervised/"
        

def dynamic_regularizer(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    penalty: bool,
    param_group_idx: int = 0
) -> None:
    """
    Dynamically adjust regularization strength based on training conditions.
    
    Args:
        optimizer: PyTorch optimizer with weight_decay parameter
        model: Neural network model containing dropout layers
        penalty: If True, increase regularization; if False, decrease it
        param_group_idx: Which parameter group to modify (default: 0)
    """
    # Adjust weight decay
    current_wd = optimizer.param_groups[param_group_idx]['weight_decay']
    
    if penalty:
        new_wd = min(current_wd * WEIGHT_DECAY_MULTIPLIER, 1.0)
    else:
        new_wd = max(current_wd / WEIGHT_DECAY_MULTIPLIER, 0.01)
    
    optimizer.param_groups[param_group_idx]['weight_decay'] = new_wd
    
    # Adjust dropout rates
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            if penalty:
                module.p = min(module.p + DROPOUT_ADJUSTMENT, 0.9)
            else:
                module.p = max(module.p - DROPOUT_ADJUSTMENT, 0.1)


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    patience: int
    lr: float
    betas: tuple[float, float]
    weight_decay: float
    accumulation_steps: int
    mask_time_prob: float


def _get_model_config(largeness: str) -> dict:
    """Get model architecture configuration based on size variant.
    
    Args:
        largeness: Model size variant ('small', 'base', or 'large')
        
    Returns:
        Dictionary containing model hyperparameters
    """
    MODEL_CONFIGS = {
        'small': {
            'hidden_size': 512,
            'num_hidden_layers': 8,
            'num_attention_heads': 8,
            'intermediate_size': 2048,
            'classifier_proj_size': 256,
            'layerdrop': 0.1,
        },
        'base': {
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'classifier_proj_size': 256,
            'layerdrop': 0.1,
        },
        'large': {
            'hidden_size': 960,
            'num_hidden_layers': 16,
            'num_attention_heads': 12,
            'intermediate_size': 3840,
            'classifier_proj_size': 512,
            'layerdrop': 0.0,
        },
    }
    
    if largeness not in MODEL_CONFIGS:
        raise ValueError(
            f"Model size '{largeness}' not supported. "
            f"Choose from: {list(MODEL_CONFIGS.keys())}"
        )
    
    return MODEL_CONFIGS[largeness]


def train(args):
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize tracking
    wandb.init(
        project="HuBert ECG",
        group="self-supervised",
        entity=None,
        name=args.wandb_run_name
    )

    # Get model configuration
    model_config = _get_model_config(args.largeness)
    
    ### configs ###
    patience = args.patience if args.patience is not None else args.training_steps // args.val_interval
    lr = args.lr
    betas = (0.9, 0.98)
    weight_decay = max(0, 0.01 * args.weight_decay_mult)
    accumulation_steps = args.accumulation_steps
    mask_time_prob = args.mask_time_prob
    
    scaler = torch.amp.GradScaler()
    

    ### START TRAINING ITERATION ###
    
    train_set = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv_train,
        #ecg_dir_path="/data/ECG_AF/train_self_supervised",
        ecg_dir_path="output/PTB",
        downsampling_factor = args.downsampling_factor,
        features_path=args.train_features_path,
        kmeans_path = args.kmeans_path,
        )

    val_set = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv_val,
        #ecg_dir_path="/data/ECG_AF/val_self_supervised",
        ecg_dir_path="output/PTB",
        features_path=args.val_features_path,
        downsampling_factor = args.downsampling_factor,
        kmeans_path = args.kmeans_path,
        )
    
    assert len(args.vocab_sizes) == train_set.ensamble_length, f"len(vocab_sizes) must be equal to the number of tasks. Found {len(args.vocab_sizes)} and {train_set.ensamble_length} tasks"
    for v, k in zip(args.vocab_sizes, train_set.ensamble_kmeans):
        assert v == k.cluster_centers_.shape[0], f"vocab_sizes must be equal to the number of clusters in the kmeans models. Found {v} and {k.cluster_centers_.shape[0]} clusters"
        
    
    train_dl = DataLoader(
        train_set,
        collate_fn=train_set.collate,
        num_workers=0,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
        )

    val_dl = DataLoader(
        val_set,
        collate_fn=val_set.collate,
        num_workers=0,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
        )

    epochs = args.training_steps // (len(train_dl) // accumulation_steps) + 1 if args.training_steps is not None else args.epochs



    if args.resume_pretraining:
        hubert_name = args.load_path.split('/')[-1]
        logger.info(f"Loading checkpoint {hubert_name} to resume pretraining")
        
        checkpoint = torch.load(args.load_path, map_location = torch.device('cpu'))

        config = checkpoint['model_config']
        assert checkpoint['pretraining_vocab_sizes'] == args.vocab_sizes
        if type(config) == HubertConfig:
            config = HuBERTECGConfig(ensemble=len(checkpoint['pretraining_vocab_sizes']), vocab_sizes=checkpoint['pretraining_vocab_sizes'], **config.to_dict())
       
        hubert = HuBERT(config)
        hubert.load_state_dict(checkpoint['model_state_dict'])

        previous_iteration = int(hubert_name.split('_')[1])

        if args.train_iteration != previous_iteration: #when switching to subsequent training iterations
            logger.info("Switching to another pretraining iteration: changing label embedding and restoring dropouts...")
            hubert.label_embedding = nn.ModuleList(nn.Embedding(vocab_size, hubert.config.classifier_proj_size) for vocab_size in args.vocab_sizes)
            
            for name, module in hubert.named_modules():
                if 'dropout' in name and 'encoder.layers' in name:
                    module.p = 0.1 # restoring p drop
                    
        # hubert = nn.DataParallel(hubert)
        hubert.to(device)
        global_step = checkpoint['global_step'] if args.train_iteration == previous_iteration else 0 
        best_val_loss = checkpoint['best_val_loss'] if args.train_iteration == previous_iteration else float('inf')
        patience_count = checkpoint['patience_count'] if args.train_iteration == previous_iteration else 0
        best_val_accuracy = checkpoint['best_val_accuracy'] if args.train_iteration == previous_iteration else 0
        
        optimizer = optim.AdamW(
            hubert.parameters(),
            lr=lr,
            betas=betas,
            eps=EPS,
            weight_decay=weight_decay,
        )
        
        if args.train_iteration == previous_iteration: #don't load state dict when switching to subsequent train iterations
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer.param_groups[0]['weight_decay'] = max(0.01, optimizer.param_groups[0]['weight_decay'])
            for name, module in hubert.named_modules():
                if 'dropout' in name:
                    module.p = max(0.1, module.p)
        
        if args.train_iteration == previous_iteration:
            lr_scheduler = get_linear_schedule_with_warmup(
               optimizer=optimizer,
               num_warmup_steps=ceil(0.08*args.training_steps  - global_step),
               num_training_steps=args.training_steps,
               last_epoch=checkpoint['lr_scheduler_state_dict']['last_epoch']-1
            )
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        else:
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=ceil(0.08*args.training_steps),
                num_training_steps=args.training_steps
            )
        
        logger.info("Checkpoint loaded.")
    else:
        logger.info("Building a model from zero to start training...")
        
        if args.downsampling_factor is None:
            conv_kernel = (10, 3, 3, 3, 3, 2, 2)
            conv_stride = (5, 2, 2, 2, 2, 2, 2)
            conv_dim = (512, 512, 512, 512, 512, 512, 512)   
        elif args.downsampling_factor == 5: 
            conv_kernel = (10, 3, 3, 2, 2)
            conv_stride = (4, 2, 2, 2, 2)
            conv_dim = (512, 512, 512, 512, 512)
        elif args.downsampling_factor == 10:
            conv_kernel = (10, 3, 3, 2)
            conv_stride = (4, 2, 2, 2)
            conv_dim = (512, 512, 512, 512)
        else:
            raise ValueError(f"Downsampling factor {args.downsampling_factor} not supported")           
            
                
        config = HuBERTECGConfig(
            ensemble_length=len(args.vocab_sizes),
            vocab_sizes=args.vocab_sizes,
            hidden_size = model_config["hidden_size"],
            num_hidden_layers = model_config["num_hidden_layers"],
            num_attention_heads = model_config["num_attention_heads"],
            intermediate_size = model_config["intermediate_size"],
            mask_time_prob = mask_time_prob, 
            classifier_proj_size = model_config["classifier_proj_size"],
            layerdrop = model_config["layerdrop"],
            conv_kernel = conv_kernel,
            conv_stride = conv_stride,
            conv_dim = conv_dim,
            mask_time_length = 1,
            hidden_dropout=max(0, 0.1 + DROPOUT_ADJUSTMENT * args.model_dropout_mult),
            activation_dropout=max(0, 0.1 + DROPOUT_ADJUSTMENT * args.model_dropout_mult),
            attention_dropout=max(0, 0.1 + DROPOUT_ADJUSTMENT * args.model_dropout_mult),
            feat_proj_dropout=max(0, 0 + DROPOUT_ADJUSTMENT * args.model_dropout_mult),
            final_dropout=max(0, 0.1 + DROPOUT_ADJUSTMENT * args.model_dropout_mult),    
        ) # + other default params
        
        hubert = HuBERT(config)
        # hubert = nn.DataParallel(hubert)
        hubert.to(device)
        global_step = 0
        best_val_loss = float("inf")
        best_val_accuracy = 0.0
        patience_count = 0        
        optimizer = optim.AdamW(
            hubert.parameters(),
            lr=lr,
            betas=betas,
            eps=EPS,
            weight_decay=weight_decay,
        )
        logger.info("Model built.")
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=ceil(0.08*args.training_steps), num_training_steps=args.training_steps)
    
    # number of params
    logger.info(f"Number of parameters: {sum(p.numel() for p in hubert.parameters())}")
    start_epoch = global_step // len(train_dl)
            
    for epoch in range(start_epoch, epochs):

        hubert.train()
        logger.info(f"Epoch {epoch+1}/{epochs}")

        train_losses = []
        
        for ecg, attention_mask, ensamble_labels in tqdm(train_dl, total=len(train_dl)):

            global_step += 1
            
            ecg = ecg.to(device) 
            attention_mask = attention_mask.to(device)
            ensamble_labels = ensamble_labels.to(device)
            
            #logger.info("Mapped data to device")

            #with amp.autocast():
            with torch.amp.autocast('cuda'):
               
                out_encoder_dict = hubert(ecg, attention_mask=attention_mask, output_attentions=False, output_hidden_states=False, return_dict=True)
                #logger.info("Computed encodings")

                mask = out_encoder_dict['mask_time_indices']
                
                ensamble_logits = hubert.logits(out_encoder_dict['last_hidden_state'])
                #logger.info("Computed logits")
                                
                # modify loss computation to enable ensamble loss (sum of losses)                
                ensamble_labels = ensamble_labels.transpose(0, 1) 
                
                masked_loss = 0
                unmasked_loss = 0
                
                assert len(ensamble_labels) == len(ensamble_logits), f"len(ensamble_labels) must be equal to len(ensamble_logits). Found {len(ensamble_labels)} and {len(ensamble_logits)}"
                
                for labels, logits in zip(ensamble_labels, ensamble_logits):
                    # labels is (BS, F), logits is (BS, F, V)
                    masked_loss += F.cross_entropy(logits[mask], labels[mask])
                    unmasked_loss += F.cross_entropy(logits[~mask], labels[~mask])
                    #logger.info("Computed masked and unmasked losses per task")
                    
                loss = args.alpha * masked_loss +  (1 - args.alpha) * unmasked_loss
                loss = loss / accumulation_steps
                       
            scaler.scale(loss).backward()
            train_losses.append(loss.item())
            
            #logger.info("Accumulated scaled loss")
            
            ### GRADIENT ACCUMULATION ###
            
            if global_step % accumulation_steps == 0:
                scaler.step(optimizer)
                lr_scheduler.step()
                scaler.update()
                optimizer.zero_grad()                

            ### VALIDATION LOOP EVERY `val_interval` STEPS + LOGGING + CHECK OF EARLY STOPPING CONDITION ###

            if global_step % args.val_interval == 0:

                hubert.eval()
                
                val_losses = []                
                val_accuracies = []
                
                logger.info(f"Validating model at step {global_step}...")
                
                ### VALIDATION LOOP ###
                
                for ecg, _, ensamble_labels in tqdm(val_dl, total=len(val_dl)):
                    ecg = (ecg).to(device)
                    #attention_mask = (attention_mask).to(device) # attention mask could harm inference performance according to HF docs
                    ensamble_labels = (ensamble_labels).to(device)
                    
                    ensamble_labels = ensamble_labels.transpose(0, 1) # (ensamble_length, BS, F)

                    with torch.no_grad():
                        out_encoder_dict = hubert(ecg, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True)
                        ensamble_logits = hubert.logits(out_encoder_dict['last_hidden_state'])
                        
                        assert len(ensamble_labels) == len(ensamble_logits), f"VAL! len(ensamble_labels) must be equal to len(ensamble_logits). Found {len(ensamble_labels)} and {len(ensamble_logits)}"
                        
                        loss = 0
                        accuracy = 0
                        for labels, logits in zip(ensamble_labels, ensamble_logits):
                            logits = logits.transpose(1, 2)
                            loss += F.cross_entropy(logits, labels)
                            accuracy += (logits.argmax(dim=1) == labels).float().mean() # mean over batch for a given task
                        
                        accuracy /= len(ensamble_logits) # mean over tasks
                        
                    val_accuracies.append(accuracy.item())                    
                    val_losses.append(loss.item())
                    
                ### END OF VALIDATION LOOP ###
                    
                val_loss = np.mean(val_losses)
                val_accuracy = np.mean(val_accuracies)
                train_loss = np.mean(train_losses)
                train_losses.clear() # to keep it aligned with validation losses
                    
                ### LOGGING ###
                
                logger.info(f"Step: {global_step}")
                logger.info(f"train_loss_{args.train_iteration}: {train_loss}")
                logger.info(f"val_loss_{args.train_iteration}: {val_loss}")
                logger.info(f"val_accuracy: {val_accuracy}")
                
                                                                        
                wandb.log({
                    f"train_loss_{args.train_iteration}" : train_loss,
                    f"val_loss_{args.train_iteration}" : val_loss,
                    "val_accuracy" : val_accuracy
                }, step=global_step)

                hubert.train()

                checkpoint_path = Path(SELF_SUPERVISED_MODEL_CKPT_PATH)
                checkpoint_path.mkdir(parents=True, exist_ok=True)

                ### SAVE IF NEW BEST MODEL + EARLY STOPPING ###
                if val_loss <= best_val_loss - MINIMAL_IMPROVEMENT: # if loss improves significantly, save checkpoint
                    
                    best_val_loss = val_loss
                    best_val_accuracy = val_accuracy if val_accuracy > best_val_accuracy else best_val_accuracy
                    patience_count = 0 
                    checkpoint = {
                                    "global_step" : global_step,
                                    "patience_count" : patience_count,
                                    "model_config" : hubert.config,
                                    "model_state_dict" : copy.deepcopy(hubert.state_dict()),
                                    "optimizer_state_dict" : copy.deepcopy(optimizer.state_dict()),
                                    "best_val_loss" : best_val_loss,
                                    "lr_scheduler_state_dict" : copy.deepcopy(lr_scheduler.state_dict()),
                                    "best_val_accuracy" : best_val_accuracy,
                                    "pretraining_vocab_sizes" : args.vocab_sizes,
                                }
                    
                    checkpoint_name = f"hubert_{args.train_iteration}_iteration_{global_step}_{wandb.run.id}.pt"
                    #torch.save(checkpoint, checkpoint_path / checkpoint_name )

                    logger.info(f"New best (best_val_loss = {best_val_loss}) - model saved at step {global_step}")
                    
                    dynamic_regularizer(optimizer, hubert, penalty=False) if args.dynamic_reg else None # unburdening model from regularization

                elif val_accuracy >= best_val_accuracy + MINIMAL_IMPROVEMENT: # if loss doesn't improve significantly but accuracy does, save checkpoint anyway
                    
                    best_val_accuracy = val_accuracy
                    checkpoint = {
                                    "global_step" : global_step,
                                    "patience_count" : patience_count,
                                    "model_config" : hubert.config,
                                    "model_state_dict" : copy.deepcopy(hubert.state_dict()),
                                    "optimizer_state_dict" : copy.deepcopy(optimizer.state_dict()),
                                    "best_val_loss" : best_val_loss,
                                    "lr_scheduler_state_dict" : copy.deepcopy(lr_scheduler.state_dict()),
                                    "best_val_accuracy" : best_val_accuracy,
                                    "pretraining_vocab_sizes" : args.vocab_sizes,
                                }
                    
                    checkpoint_name = f"hubert_{args.train_iteration}_iteration_{global_step}_{wandb.run.id}.pt"                                          
                    
                    #torch.save(checkpoint, checkpoint_path / checkpoint_name)
                    logger.info(f"Val loss not improved but val accuracy did (best_val_accuracy = {best_val_accuracy}) - model saved at step {global_step}")   
                    
                    dynamic_regularizer(optimizer, hubert, penalty=False) if args.dynamic_reg else None # unburdening model from regularization
                    
                else: #worsening performance
                    patience_count += 1
                     
                    if args.dynamic_reg and patience_count % (patience // args.intervals_for_penalty) == 0 and patience_count != patience:
                        dynamic_regularizer(optimizer, hubert, penalty=True) # penalizing model with regularization
                    
                    if patience_count == patience:
                        logger.warning(f"EARLY STOPPING: Max num of val intervals with no improvement reached at {global_step}")
                        wandb.log({
                            "patience_count" : patience_count
                        })
                        return
                    

    ### END OF TRAINING ITERATION ###
    logger.info("End of training")
    logger.info(f"STATS: Global step={global_step}, Best val loss={best_val_loss}")
    wandb.finish()


def main() -> None:
    args = create_training_parser()
    init_seeds(seed=42)
    train(args)


if __name__ == "__main__":
    main()
