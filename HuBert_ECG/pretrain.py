import copy
import logging
import torch
import wandb

import numpy as np
import torch.nn as nn
import torch.optim as optim

from dataclasses import dataclass
from loguru import logger
from math import ceil
from pathlib import Path
from rich.logging import RichHandler
from tqdm import tqdm
from transformers import HubertConfig
from transformers import get_linear_schedule_with_warmup
# from transformers.models.hubert.modeling_hubert import compute_mask_indices

from torch.nn import functional as F

# Import custom modules
from dataset import create_dataloader
from config import create_training_parser, init_seeds
from hubert_ecg import HuBERTECG as HuBERT, HuBERTECGConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", 
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

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


def _validate_vocab_sizes(args, dataset):
    """Validate vocabulary sizes match k-means cluster counts."""
    assert len(args.vocab_sizes) == dataset.ensamble_length, (
        f"Number of vocab_sizes ({len(args.vocab_sizes)}) must match "
        f"number of tasks ({dataset.ensamble_length})"
    )
    
    for vocab_size, kmeans in zip(args.vocab_sizes, dataset.ensamble_kmeans):
        n_clusters = kmeans.cluster_centers_.shape[0]
        assert vocab_size == n_clusters, (
            f"vocab_size ({vocab_size}) must match number of k-means "
            f"clusters ({n_clusters})"
        )


def validate_model(model, val_loader, device, logger, global_step):
    """Validation loop extracted as separate function."""
    model.eval()
    
    val_losses = []
    val_accuracies = []
    
    logger.info(f"Validating model at step {global_step}...")
    
    with torch.no_grad():
        for ecg, _, ensemble_labels in tqdm(val_loader, total=len(val_loader)):
            ecg, ensemble_labels = ecg.to(device), ensemble_labels.to(device)
            # attention_mask = (attention_mask).to(device) # attention mask could harm inference performance according to HF docs

            # (ensamble_length, batch_size, sequence_length)
            ensemble_labels = ensemble_labels.transpose(0, 1)

            # Forward pass (no attention mask during validation per HF recommendation)
            encoder_output = model(
                ecg, 
                attention_mask=None, 
                output_attentions=False, 
                output_hidden_states=False, 
                return_dict=True
            )
            ensemble_logits = model.logits(encoder_output['last_hidden_state'])

            # labels: (batch_size, sequence_length), logits: (batch_size, sequence_length, vocab_size)
            assert len(ensemble_labels) == len(ensemble_logits), f"VAL! len(ensamble_labels) must be equal to len(ensamble_logits). Found {len(ensemble_labels)} and {len(ensemble_logits)}"

            # Compute loss and accuracy across ensemble
            batch_loss = 0
            batch_accuracy = 0

            for labels, logits in zip(ensemble_labels, ensemble_logits):
                logits_transposed = logits.transpose(1, 2)
                batch_loss += F.cross_entropy(logits_transposed, labels)
                batch_accuracy += (logits_transposed.argmax(dim=1) == labels).float().mean()
            
            batch_accuracy /= len(ensemble_logits)
            
            val_losses.append(batch_loss.item())
            val_accuracies.append(batch_accuracy.item())
    
    return np.mean(val_losses), np.mean(val_accuracies)


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
    
    # Setup data
    train_loader = create_dataloader(
        csv_path=args.path_to_dataset_csv_train,
        ecg_dir=args.ecg_dir_path,
        batch_size=args.batch_size,
        downsample_factor=args.downsampling_factor,
        features_path=args.train_features_path,
        kmeans_path=args.kmeans_path,
        is_pretrain=True,
        drop_last=False
    )
    val_loader = create_dataloader(
        csv_path=args.path_to_dataset_csv_val,
        ecg_dir=args.ecg_dir_path,
        batch_size=args.batch_size,
        downsample_factor=args.downsampling_factor,
        features_path=args.val_features_path,
        kmeans_path=args.kmeans_path,
        is_pretrain=True,
        shuffle=False,
        drop_last=False
    )

    # Validate configuration
    _validate_vocab_sizes(args, train_loader.dataset)

    if args.training_steps is not None:
        steps_per_epoch = len(train_loader) // accumulation_steps
        epochs =  args.training_steps // steps_per_epoch + 1
    else:
        epochs = args.epochs

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
    start_epoch = global_step // len(train_loader)

    for epoch in range(start_epoch, epochs):

        hubert.train()
        logger.info(f"Epoch {epoch+1}/{epochs}")

        train_losses = []
        
        for ecg, attention_mask, ensemble_labels in tqdm(train_loader, total=len(train_loader)):

            global_step += 1
            
            ecg = ecg.to(device) 
            attention_mask = attention_mask.to(device)
            ensemble_labels = ensemble_labels.to(device)
            
            #logger.info("Mapped data to device")

            #with amp.autocast():
            with torch.amp.autocast('cuda'):
               
                out_encoder_dict = hubert(ecg, attention_mask=attention_mask, output_attentions=False, output_hidden_states=False, return_dict=True)
                #logger.info("Computed encodings")

                mask = out_encoder_dict['mask_time_indices']
                
                ensemble_logits = hubert.logits(out_encoder_dict['last_hidden_state'])
                #logger.info("Computed logits")
                                
                # modify loss computation to enable ensamble loss (sum of losses)                
                ensemble_labels = ensemble_labels.transpose(0, 1) 
                
                masked_loss = 0
                unmasked_loss = 0
                
                assert len(ensemble_labels) == len(ensemble_logits), f"len(ensamble_labels) must be equal to len(ensamble_logits). Found {len(ensemble_labels)} and {len(ensemble_logits)}"
                
                for labels, logits in zip(ensemble_labels, ensemble_logits):
                    # labels is (BS, F), logits is (BS, F, V)
                    masked_loss += F.cross_entropy(logits[mask], labels[mask])
                    unmasked_loss += F.cross_entropy(logits[~mask], labels[~mask])
                    #logger.info("Computed masked and unmasked losses per task")
                    
                loss = args.alpha * masked_loss +  (1 - args.alpha) * unmasked_loss
                loss = loss / accumulation_steps
                       
            scaler.scale(loss).backward()
            train_losses.append(loss.item())
            
            #logger.info("Accumulated scaled loss")
            
            # Gradient accumulation
            if global_step % accumulation_steps == 0:
                scaler.step(optimizer)
                lr_scheduler.step()
                scaler.update()
                optimizer.zero_grad()


            # Validation every val_interval steps
            if global_step % args.val_interval == 0:
                val_loss, val_accuracy = validate_model(
                    hubert, val_loader, device, logger, global_step
                )
                
                train_loss = np.mean(train_losses)
                train_losses.clear()

                # Logging
                logger.info(f"Step: {global_step}")
                logger.info(f"train_loss: {train_loss}")
                logger.info(f"val_loss: {val_loss}")
                logger.info(f"val_accuracy: {val_accuracy}")
                
                                                                        
                wandb.log({
                    f"train_loss": train_loss,
                    f"val_loss": val_loss,
                    "val_accuracy": val_accuracy
                }, step=global_step)

                hubert.train()

                checkpoint_path = Path(SELF_SUPERVISED_MODEL_CKPT_PATH)
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                            
                # Save if new best model + Early stopping
                loss_improved = val_loss <= best_val_loss - MINIMAL_IMPROVEMENT
                accuracy_improved = val_accuracy >= best_val_accuracy + MINIMAL_IMPROVEMENT
                
                if loss_improved or accuracy_improved:
                    # Update best metrics
                    if loss_improved:
                        best_val_loss = val_loss
                        best_val_accuracy = max(val_accuracy, best_val_accuracy)
                        patience_count = 0
                        logger.info(
                            f"New best (best_val_loss={best_val_loss:.4f}) - "
                            f"model saved at step {global_step}"
                        )
                    else:  # accuracy_improved only
                        best_val_accuracy = val_accuracy
                        logger.info(
                            f"Val loss not improved but val accuracy did "
                            f"(best_val_accuracy={best_val_accuracy:.4f}) - "
                            f"model saved at step {global_step}"
                        )
                    
                    # Create checkpoint
                    checkpoint = {
                        "global_step": global_step,
                        "patience_count": patience_count,
                        "model_config": hubert.config,
                        "model_state_dict": copy.deepcopy(hubert.state_dict()),
                        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                        "lr_scheduler_state_dict": copy.deepcopy(lr_scheduler.state_dict()),
                        "best_val_loss": best_val_loss,
                        "best_val_accuracy": best_val_accuracy,
                        "pretraining_vocab_sizes": args.vocab_sizes,
                    }
                    
                    checkpoint_name = (
                        f"hubert_{args.train_iteration}_iteration_"
                        f"{global_step}_{wandb.run.id}.pt"
                    )
                    # torch.save(checkpoint, checkpoint_path / checkpoint_name)
                    
                    # Reduce regularization after improvement
                    if args.dynamic_reg:
                        dynamic_regularizer(optimizer, hubert, penalty=False)
                
                else:  # No improvement
                    patience_count += 1
                    
                    # Apply regularization penalty at intervals
                    if args.dynamic_reg and patience_count != patience:
                        if patience_count % (patience // args.intervals_for_penalty) == 0:
                            dynamic_regularizer(optimizer, hubert, penalty=True)
                    
                    # Early stopping
                    if patience_count >= patience:
                        logger.warning(
                            f"EARLY STOPPING: Max patience reached at step {global_step} "
                            f"(patience_count={patience_count})"
                        )
                        return

    # End of training iteration
    logger.info("End of training")
    logger.info(f"STATS: Global step={global_step}, Best val loss={best_val_loss:.4f}")
    wandb.finish()


def main() -> None:
    args = create_training_parser()
    init_seeds(seed=42)
    train(args)


if __name__ == "__main__":
    main()
