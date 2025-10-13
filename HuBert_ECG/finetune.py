import copy
import logging
import os
import torch
import wandb

import numpy as np
import torch.nn as nn
import torch.optim as optim

from math import ceil
from pathlib import Path
from rich.logging import RichHandler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HubertConfig
from transformers import get_linear_schedule_with_warmup

# Import custom modules
# from trainer import Trainer
from logging_utils import ClearMLLogger
from validator import Validator
from metricsV2 import FinetuneMetrics
from config import create_parser, init_seeds
from dataset import ECGDataset
from hubert_ecg import HuBERTECG as HuBERT, HuBERTECGConfig
from hubert_ecg_classification import HuBERTForECGClassification as HuBERTClassification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", 
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

EPS = 1e-9
MINIMAL_IMPROVEMENT = 1e-3
DROPOUT_ADJUSTMENT = 0.05
WEIGHT_DECAY_MULTIPLIER = 5.0


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


def create_dataloader(
    csv_path: str,
    ecg_dir: str,
    batch_size: int,
    label_start_idx: int = 3,
    downsample_factor: int = None,
    random_crop: bool = False,
    shuffle: bool = True,
    is_pretrain: bool = False,
) -> DataLoader:
    """Create a DataLoader for ECG dataset.
    
    Args:
        csv_path: Path to dataset CSV file
        ecg_dir: Directory containing ECG data
        batch_size: Batch size for DataLoader
        label_start_idx: Starting index of labels in CSV
        downsample_factor: Factor for downsampling ECG signals
        random_crop: Whether to apply random 5s crop augmentation
        shuffle: Whether to shuffle data
        is_pretrain: Whether this is for pretraining mode

    Returns:
        Configured DataLoader instance
    """
    dataset = ECGDataset(
        path_to_dataset_csv=csv_path,
        ecg_dir_path=ecg_dir,
        label_start_index=label_start_idx,
        downsampling_factor=downsample_factor,
        pretrain=is_pretrain,
        random_crop=random_crop,
    )

    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty! No images found. Please check the paths and file formats.")
    
    data_loader = DataLoader(
        dataset,
        collate_fn=dataset.collate,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset samples: {len(dataset)}, DataLoader batches: {len(data_loader)}")

    return data_loader

class CheckpointManager:
    """Handles model checkpointing."""
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(f"{save_dir}/checkpoints")
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        model,
        optimizer,
        lr_scheduler,
        global_step: int,
        best_val_loss: float,
        patience_count: int,
        args,
        target_score: float,
        run_id: str,
    ):
        """Save model checkpoint."""
        checkpoint = {
            'global_step': global_step,
            'best_val_loss': best_val_loss,
            'model_config': model.config,
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
            'lr_scheduler_state_dict': copy.deepcopy(lr_scheduler.state_dict()),
            'patience_count': patience_count,
            'linear': args.classifier_hidden_size is None and not args.use_label_embedding,
            'finetuning_vocab_size': args.vocab_size,
            'use_label_embedding': args.use_label_embedding,
            f'target_val_{args.target_metric}': target_score,
        }

        checkpoint_name = f"hubert_{args.train_iteration}_iteration_{global_step}_finetuned_{run_id}.pt"
        save_path = self.save_dir / checkpoint_name
        torch.save(checkpoint, save_path)
        
        return save_path
    

def finetune(args):

    clearml_config = {
        'project': "HuBERT-ECG",  # ClearML project name
        'task_name': "HuBERT-ECG finetune",  
        'task_type': "training",  # ClearML task type
        'reuse_last_task_id': False,  # ClearML task ID to resume or boolean flag
        "tags": [args.task],  # List of tags for ClearML task
    }

    # ClearML uses 1337 as the default initial seed
    clearml_logger = ClearMLLogger(args.output_dir, **clearml_config)

    # Upload args as JSON
    clearml_logger.log_args_as_json(args)

    init_seeds()

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(args.output_dir)

    device = torch.device('cuda')
    
    wandb.init(project="my-project", group="supervised", entity=None)

    if args.wandb_run_name is not None:
        wandb.run.name = args.wandb_run_name

    train_loader = create_dataloader(
        csv_path=args.path_to_dataset_csv_train,
        ecg_dir=args.ecg_dir,
        batch_size=args.batch_size,
        label_start_idx=args.label_start_index,
        downsample_factor=args.downsampling_factor,
        random_crop=args.random_crop,
        shuffle=True,
    )
    
    val_loader = create_dataloader(
        csv_path=args.path_to_dataset_csv_val,
        ecg_dir=args.ecg_dir,
        batch_size=args.batch_size,
        label_start_idx=args.label_start_index,
        downsample_factor=args.downsampling_factor,
        random_crop=args.random_crop,
        shuffle=False,
    )
    
    train_pos_weights = train_loader.dataset.weights.to(device) if args.use_loss_weights else None
    val_pos_weights = val_loader.dataset.weights.to(device) if args.use_loss_weights else None
    

    lr = args.lr
    betas = (0.9, 0.98)
    weight_decay = max(0, 0.01 * args.weight_decay_mult)
    accumulation_steps = args.accumulation_steps
    
    task2criteria = {
        'multi_class': (torch.nn.CrossEntropyLoss(weight=train_pos_weights), torch.nn.CrossEntropyLoss(weight=val_pos_weights)),
        'multi_label': (torch.nn.BCEWithLogitsLoss(pos_weight=train_pos_weights), torch.nn.BCEWithLogitsLoss(pos_weight=val_pos_weights)),
        'regression' : (torch.nn.MSELoss(), torch.nn.MSELoss())
    }    
    
    criterion = task2criteria[args.task]
    criterion_train = criterion[0].to(device)
    criterion_val = criterion[1].to(device)
    
    args.training_steps = args.training_steps if args.training_steps is not None else ((args.epochs - 1) * (len(train_loader) // accumulation_steps))
    
    args.val_interval = len(train_loader) if args.val_interval is None else args.val_interval
    
    logger.info(f"{args.training_steps} training steps to perform")
    logger.info(f"{args.val_interval} steps to wait before validation")

    if args.resume_finetuning:
        
        logger.info(f"Loading pretraining checkpoint {args.load_path.split('/')[-1]} to resume finetuning")
        
        checkpoint = torch.load(args.load_path, map_location = 'cpu', weights_only=False)

        config = checkpoint['model_config']

        if type(config) == HubertConfig:
            config = HuBERTECGConfig(**config.to_dict())
        config.conv_pos_batch_norm = False
        
        pretrained_hubert = HuBERT(config)
        
        assert checkpoint['finetuning_vocab_size'] == args.vocab_size, "Vocab size mismatch"
        assert checkpoint['use_label_embedding'] == args.use_label_embedding, "Label embedding mismatch"
        assert checkpoint['linear'] == True if args.classifier_hidden_size is None and not args.use_label_embedding else False, "Classifier mismatch"
        
        hubert = HuBERTClassification(pretrained_hubert, num_labels=args.finetuning_vocab_size, classifier_hidden_size=args.classifier_hidden_size, use_label_embedding=args.use_label_embedding)
        hubert.to(device)
        
        hubert.load_state_dict(checkpoint['model_state_dict'], strict=False) # strict false prevents errors when trying to match mask token key
        
        global_step = checkpoint['global_step']
        best_val_loss = checkpoint['best_val_loss']
        patience_count = checkpoint['patience_count']
        best_val_target_score = checkpoint[f'target_val_{args.target_metric}']
        
        if args.freezing_steps is not None and global_step < args.freezing_steps:
            hubert.set_transformer_blocks_trainable(n_blocks=0)
            hubert.set_feature_extractor_trainable(False)
        else:
            hubert.set_transformer_blocks_trainable(n_blocks=args.transformer_blocks_to_unfreeze)
            hubert.set_feature_extractor_trainable(args.unfreeze_conv_embedder)
            
        # if layuer_wise_lr, then set a higher lr for deeper transformer layers + head than that of the rest of the trainable body of the model
        parameters_group = []    
        if args.layer_wise_lr and all(p.requires_grad for p in hubert.hubert_ecg.encoder.layers.parameters()):
            parameters_group.append({"params": hubert.hubert_ecg.feature_projection.parameters(), "lr": 1e-7})
            parameters_group.append({"params": hubert.hubert_ecg.encoder.layers[:args.transformer_blocks_to_unfreeze-4].parameters(), "lr": 1e-7})
            parameters_group.append({"params": hubert.hubert_ecg.encoder.layers[args.transformer_blocks_to_unfreeze-4:].parameters(), "lr": lr})
            parameters_group.append({"params": hubert.classifier.parameters(), "lr": 1e-5})
        else:
            parameters_group.append({"params" : filter(lambda p : p.requires_grad, hubert.parameters()), "lr": lr})
        
        optimizer = optim.AdamW(
            parameters_group,
            betas=betas,
            eps=EPS,
            weight_decay=weight_decay,
        )

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=ceil(args.ramp_up_perc*(args.training_steps-global_step)),
            num_training_steps=args.training_steps,
        )

        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        logger.info("Checkpoint loaded. Model ready to resume finetuning.")
        
    elif args.random_init:
        logger.info("Creating a model for fully supervised training")
        
        ### size model hyperparams ###
        if args.largeness == "base":
            hidden_size = 768
            num_hidden_layers = 12
            num_attention_heads = 12
            intermediate_size = 3072    
            classifier_proj_size = 256
        elif args.largeness == "large":
            hidden_size = 960
            num_hidden_layers = 16
            num_attention_heads = 12
            intermediate_size = 3840
            classifier_proj_size = 512
        elif args.largeness == 'small': # small
            hidden_size = 512
            num_hidden_layers = 8
            num_attention_heads = 8
            intermediate_size = 2048
            classifier_proj_size = 256
        else:
            raise ValueError(f"Model largeness {args.largeness} not supported")
        
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
            hidden_size = hidden_size,
            num_hidden_layers = num_hidden_layers,
            num_attention_heads = num_attention_heads,
            intermediate_size = intermediate_size,
            mask_time_prob = 0.0, 
            classifier_proj_size = classifier_proj_size,
            layerdrop = args.finetuning_layerdrop,
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
        
        hubert = HuBERTClassification(hubert,
                                      num_labels=args.vocab_size,
                                      classifier_hidden_size=args.classifier_hidden_size,
                                      use_label_embedding=args.use_label_embedding
                                      )
        hubert.to(device)  
        
        
        global_step = 0
        best_val_loss = float("inf")
        best_val_target_score = 0.0
        patience_count = 0  
              
        optimizer = optim.AdamW(
            hubert.parameters(),
            lr=lr,
            betas=betas,
            eps=EPS,
            weight_decay=weight_decay,
        )
        
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=ceil(0.08*args.training_steps),
            num_training_steps=args.training_steps
        )
        
        logger.info("Model created. Ready for fully supervised training.")
        
    else:
        logger.info(f"Loading pretraining checkpoint {args.load_path.split('/')[-1]} to start finetuning")
                
        checkpoint = torch.load(args.load_path, map_location = 'cpu', weights_only=False)
        config = checkpoint['model_config']
        if type(config) == HubertConfig:
            config = HuBERTECGConfig(**config.to_dict())
        config.layerdrop = args.finetuning_layerdrop
        config.conv_pos_batch_norm = False

        pretrained_hubert = HuBERT(config)

        # restore original p-dropout or set multipliers
        for name, module in pretrained_hubert.named_modules():
            if 'dropout' in name:
                module.p = 0.1 + DROPOUT_ADJUSTMENT * args.model_dropout_mult
        
        hubert = HuBERTClassification(pretrained_hubert, num_labels=args.vocab_size, classifier_hidden_size=args.classifier_hidden_size,  use_label_embedding=args.use_label_embedding)
        hubert.to(device)         

        # Transfer learning: Loading a pretrained model but with a different final layer
        hubert.hubert_ecg.load_state_dict(checkpoint['model_state_dict'], strict=False) # load backbone weights

        global_step = 0
        best_val_loss = float('inf')
        patience_count = 0
        best_val_target_score = 0.0
        
        
        # ASSUMPTION: the classification head is always trainable
        # transformer_blocks_to_unfreeze is 0 by default, meaning that if no input comes from the user, the transformer encoder remains frozen
        # if the user wants to unfreeze some blocks, he must provide the number of blocks to unfreeze.
        
        # Freezing the convolutional feature extractor is a precise choice of the user. By default it is frozen, but it can unfrozen by using the flag --unfreeze_conv_embedder
        
        # If the freezing_steps argument is provided, then the model is completely frozen to allow to train only the head for that number of steps.
        # After that, n blocks of the transformer's encoder are unfronzen and the conv feature extractor is unfrozen as well in the training loop if the flag --unfreeze_conv_embedder is used.
        # Freezing_steps is none by default
        
        # Nte: if freezing_steps equals training_steps, then the model is completely frozen for the whole training
        
        if args.freezing_steps is not None:
            hubert.set_transformer_blocks_trainable(n_blocks=0) # freeze all transformer blocks
            hubert.set_feature_extractor_trainable(False) # freeze conv feature extractor
        else:        
            hubert.set_transformer_blocks_trainable(n_blocks=args.transformer_blocks_to_unfreeze) # makes trainable only the last n_blocks of the transformer encoder
            hubert.set_feature_extractor_trainable(args.unfreeze_conv_embedder) # frozen by default
        
        # if layer_wise_lr, then set a higher lr for deeper transformer layers + head than that of the rest of the trainable body of the model
        # first 8 layer with lower lr and last 4 with higher lr based on Primer Bertology
        parameters_group = []    
        if args.layer_wise_lr and all(p.requires_grad for p in hubert.hubert_ecg.encoder.layers.parameters()):
            logger.info("Setting layer-wise learning rate")
            parameters_group.append({"params": hubert.hubert_ecg.feature_projection.parameters(), "lr": 1e-7})
            parameters_group.append({"params": hubert.hubert_ecg.encoder.layers[:args.transformer_blocks_to_unfreeze-4].parameters(), "lr": 1e-7})
            parameters_group.append({"params": hubert.hubert_ecg.encoder.layers[args.transformer_blocks_to_unfreeze-4:].parameters(), "lr": lr})
            parameters_group.append({"params": hubert.classifier.parameters(), "lr": 1e-5})
        else:
            parameters_group.append({"params" : filter(lambda p : p.requires_grad, hubert.parameters()), "lr": lr})
        
        optimizer = optim.AdamW(
            parameters_group,
            betas=betas,
            eps=EPS,
            weight_decay=weight_decay,
        )
        
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=ceil(args.ramp_up_perc * args.training_steps),
            num_training_steps=args.training_steps,
        )
        
        logger.info("Checkpoint loaded. Model ready for finetuning.")
    
    # ...
    pass
    # Ignore the previous code


    scaler = torch.amp.GradScaler('cuda') 

    epochs = args.training_steps // (len(train_loader) // accumulation_steps) + 1 if args.training_steps is not None else args.epochs
    
    start_epoch = global_step // len(train_loader)

    # Initialize metrics tracker
    val_metrics = FinetuneMetrics(
        task=args.task,
        num_labels=args.vocab_size,
        split='val'
    ).to(device)

    # Initialize validator
    validator = Validator(
        model=hubert,
        val_loader=val_loader,
        criterion=criterion_val,
        metrics=val_metrics,
        device=device,
        target_metric=args.target_metric
    )
    
    # Validate that target metric is available
    if args.target_metric not in val_metrics.metrics.keys():
        raise ValueError(f"Target metric {args.target_metric} not available for task {args.task}")
    
    for epoch in range(start_epoch, epochs):
    
        hubert.train()
        train_losses = []
        
        for ecg, attention_mask, labels in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}"):
            
            global_step += 1
            
            ecg = ecg.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.squeeze().to(device)
            
            with torch.amp.autocast('cuda'):
                logits, _ = hubert(ecg, attention_mask=attention_mask, output_attentions=False, output_hidden_states=False, return_dict=False)
                loss = criterion_train(logits, labels)
                loss /= accumulation_steps  # Normalize loss
                
            scaler.scale(loss).backward() # accumulate normalized loss
            train_losses.append(loss.item())                
                
            if global_step % accumulation_steps == 0: 
                scaler.step(optimizer)
                lr_scheduler.step()
                scaler.update()
                optimizer.zero_grad()
                
            if args.freezing_steps is not None and global_step >= args.freezing_steps:
                
                # Unfreeze what you wanted to unfreeze, as specified by user input
                hubert.set_transformer_blocks_trainable(n_blocks=args.transformer_blocks_to_unfreeze)
                hubert.set_feature_extractor_trainable(args.unfreeze_conv_embedder)
                
                # recreate optimizer to update params when unfreezing them
                optimizer = optim.AdamW(
                    filter(lambda p: p.requires_grad, hubert.parameters()),
                    lr=lr,
                    betas=betas,
                    eps=EPS,
                    weight_decay=weight_decay,
                )
                # recreate lr_scheduler to update params when unfreezing them
                lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=ceil(args.ramp_up_perc*(args.training_steps-global_step)),
                    num_training_steps=args.training_steps,
                )
            
            ### validation ###        
            if global_step % args.val_interval == 0:
                # Run validation using the Validator
                results = validator.validate()

                # Extract metrics
                val_loss = results['val_loss']
                target_score = results['target_score']
                
                train_loss = np.mean(train_losses)
                train_losses.clear()  # to keep train loss aligned with val loss
                
                # Log metrics to wandb
                wandb.log({"Train_loss": train_loss, "val_loss": val_loss}, step=global_step)

                for metric_name, metric_value in results.items():
                    if metric_name != 'target_score' and '_macro' in metric_name:
                        wandb.log({metric_name: metric_value}, step=global_step)

                results.update({"Train_loss": train_loss, 
                                "learning_rate": optimizer.param_groups[0]['lr']})
                
                clearml_logger.log_metrics(results, global_step)

                hubert.train()
                
                ### save if there's improvement in loss or target metric ###

                # Checkpoint logic
                should_save = False
                reason = ""

                if val_loss <= best_val_loss - MINIMAL_IMPROVEMENT:
                    best_val_loss = val_loss
                    patience_count = 0
                    should_save = True
                    reason = f"New best val loss = {best_val_loss:.4f}"

                    # Reward for loss
                    if args.dynamic_reg:
                        dynamic_regularizer(optimizer, hubert, penalty=False)

                elif target_score >= best_val_target_score + MINIMAL_IMPROVEMENT:
                    best_val_target_score = target_score
                    should_save = True
                    reason = f"Val loss not improved but {args.target_metric} improved (= {target_score:.4f})"
                    
                    # Reward for target metric
                    if args.dynamic_reg:
                        dynamic_regularizer(optimizer, hubert, penalty=False)

                # Loss not improved and target metric not improved
                else:
                    patience_count += 1
                    
                    if args.dynamic_reg:
                        penalty_interval = args.patience // args.intervals_for_penalty
                        if patience_count % penalty_interval == 0 and patience_count != args.patience:
                            dynamic_regularizer(optimizer, hubert, penalty=True)
                    
                    if patience_count >= args.patience:
                        logger.warning(f"Early stopping at step {global_step}")
                        wandb.log({"patience_count": patience_count})
                        return
                
                if should_save:
                    checkpoint_manager.save(
                        hubert, optimizer, lr_scheduler,
                        global_step, best_val_loss, patience_count,
                        args, target_score, wandb.run.id
                    )
                    logger.info(f"{reason}. Checkpoint saved at step {global_step}")

    # Training completed
    logger.info("End of finetuning.")
    logger.info(f"Best val loss = {best_val_loss:.4f}")
    logger.info(f"Best val {args.target_metric} = {best_val_target_score:.4f}")


if __name__ == "__main__":
    args = create_parser()

    # Start training
    finetune(args)
