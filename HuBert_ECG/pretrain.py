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
        
# Constants
WARMUP_RATIO = 0.08
DROPOUT_RESET_VALUE = 0.1
MIN_WEIGHT_DECAY = 0.01

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

            assert len(ensemble_labels) == len(ensemble_logits), f"VAL! len(ensamble_labels) must be equal to len(ensamble_logits). Found {len(ensemble_labels)} and {len(ensemble_logits)}"

            # Compute loss and accuracy across ensemble
            batch_loss = 0
            batch_accuracy = 0

            for labels, logits in zip(ensemble_labels, ensemble_logits):
                # labels: (batch_size, seq_len), logits: (batch_size, seq_len, vocab_size)
                logits_transposed = logits.transpose(1, 2)
                batch_loss += F.cross_entropy(logits_transposed, labels)
                batch_accuracy += (logits_transposed.argmax(dim=1) == labels).float().mean()
            
            batch_accuracy /= len(ensemble_logits)
            
            val_losses.append(batch_loss.item())
            val_accuracies.append(batch_accuracy.item())
    
    return np.mean(val_losses), np.mean(val_accuracies)


def _resume_from_checkpoint(args, device):
    """Load model and training state from checkpoint."""
    checkpoint_name = args.load_path.split('/')[-1]
    logger.info(f"Loading checkpoint {checkpoint_name} to resume pretraining")
    
    checkpoint = torch.load(args.load_path, map_location=torch.device('cpu'), weights_only=False)
    
    # Validate checkpoint
    assert checkpoint['pretraining_vocab_sizes'] == args.vocab_sizes, \
        "Vocab sizes mismatch between checkpoint and args"

    # # Create config
    # config = _create_config_from_checkpoint(checkpoint, args.vocab_sizes)
    config = checkpoint['model_config']

    # Initialize and load model
    model = HuBERT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    previous_iteration = int(checkpoint_name.split('_')[1])
    is_same_iteration = (args.train_iteration == previous_iteration)
    
    # Handle iteration switch
    if not is_same_iteration:
        logger.info("Switching to another pretraining iteration: "
                   "reinitializing label embedding and restoring dropouts...")
        _reset_label_embedding(model, args.vocab_sizes)
        _reset_encoder_dropouts(model, DROPOUT_RESET_VALUE)
    
    model.to(device)
    
    # Prepare training state
    training_state = {
        'global_step': checkpoint['global_step'] if is_same_iteration else 0,
        'best_val_loss': checkpoint['best_val_loss'] if is_same_iteration else float('inf'),
        'patience_count': checkpoint['patience_count'] if is_same_iteration else 0,
        'best_val_accuracy': checkpoint['best_val_accuracy'] if is_same_iteration else 0.0,
        'is_same_iteration': is_same_iteration,
        'optimizer_state': checkpoint.get('optimizer_state_dict'),
        'lr_scheduler_state': checkpoint.get('lr_scheduler_state_dict') if is_same_iteration else None
    }
    
    logger.info("Checkpoint loaded.")
    return model, training_state


def _initialize_model_from_scratch(args, model_config, mask_time_prob, device):
    """Initialize model from scratch with given configuration."""
    logger.info("Building a model from zero to start training...")
    
    conv_configs = _get_conv_config(args.downsampling_factor)
    
    config = HuBERTECGConfig(
        ensemble_length=len(args.vocab_sizes),
        vocab_sizes=args.vocab_sizes,
        hidden_size=model_config["hidden_size"],
        num_hidden_layers=model_config["num_hidden_layers"],
        num_attention_heads=model_config["num_attention_heads"],
        intermediate_size=model_config["intermediate_size"],
        mask_time_prob=mask_time_prob,
        classifier_proj_size=model_config["classifier_proj_size"],
        layerdrop=model_config["layerdrop"],
        mask_time_length=1,
        hidden_dropout=max(0, 0.1 + DROPOUT_ADJUSTMENT * args.model_dropout_mult),
        activation_dropout=max(0, 0.1 + DROPOUT_ADJUSTMENT * args.model_dropout_mult),
        attention_dropout=max(0, 0.1 + DROPOUT_ADJUSTMENT * args.model_dropout_mult),
        feat_proj_dropout=max(0, 0 + DROPOUT_ADJUSTMENT * args.model_dropout_mult),
        final_dropout=max(0, 0.1 + DROPOUT_ADJUSTMENT * args.model_dropout_mult),
        **conv_configs
    )

    # model = nn.DataParallel(model)
    model = HuBERT(config)
    model.to(device)
    
    training_state = {
        'global_step': 0,
        'best_val_loss': float('inf'),
        'best_val_accuracy': 0.0,
        'patience_count': 0,
        'is_same_iteration': False,
        'optimizer_state': None,
        'lr_scheduler_state': None
    }
    
    logger.info("Model built.")
    return model, training_state


def _get_conv_config(downsampling_factor):
    """Get convolutional layer configuration based on downsampling factor."""
    configs = {
        None: {
            'conv_kernel': (10, 3, 3, 3, 3, 2, 2),
            'conv_stride': (5, 2, 2, 2, 2, 2, 2),
            'conv_dim': (512, 512, 512, 512, 512, 512, 512)
        },
        5: {
            'conv_kernel': (10, 3, 3, 2, 2),
            'conv_stride': (4, 2, 2, 2, 2),
            'conv_dim': (512, 512, 512, 512, 512)
        },
        10: {
            'conv_kernel': (10, 3, 3, 2),
            'conv_stride': (4, 2, 2, 2),
            'conv_dim': (512, 512, 512, 512)
        }
    }
    
    if downsampling_factor not in configs:
        raise ValueError(f"Downsampling factor {downsampling_factor} not supported. "
                        f"Supported values: {list(configs.keys())}")
    
    return configs[downsampling_factor]


def _create_config_from_checkpoint(checkpoint, vocab_sizes):
    """Create appropriate config from checkpoint."""
    config = checkpoint['model_config']
    
    if isinstance(config, HubertConfig):
        config = HuBERTECGConfig(
            ensemble=len(checkpoint['pretraining_vocab_sizes']),
            vocab_sizes=checkpoint['pretraining_vocab_sizes'],
            **config.to_dict()
        )
    return config


def _reset_label_embedding(model, vocab_sizes):
    """Reinitialize label embeddings for new training iteration."""
    model.label_embedding = nn.ModuleList([
        nn.Embedding(vocab_size, model.config.classifier_proj_size)
        for vocab_size in vocab_sizes
    ])


def _reset_encoder_dropouts(model, dropout_value):
    """Reset dropout rates for encoder layers."""
    for name, module in model.named_modules():
        if 'dropout' in name and 'encoder.layers' in name:
            module.p = dropout_value


def _ensure_min_dropout(model, min_dropout):
    """Ensure all dropout modules have at least the minimum dropout rate."""
    for name, module in model.named_modules():
        if 'dropout' in name:
            module.p = max(min_dropout, module.p)


def _create_optimizer(model, lr, betas, weight_decay):
    """Create AdamW optimizer with specified parameters."""
    return optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=EPS,
        weight_decay=weight_decay
    )


def _create_lr_scheduler(optimizer, total_steps, warmup_ratio, 
                         current_step=0, previous_state=None):
    """Create learning rate scheduler with warmup."""
    num_warmup_steps = ceil(warmup_ratio * total_steps)
    
    if previous_state is not None:
        # Resume from previous scheduler state
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps - current_step,
            num_training_steps=total_steps,
            last_epoch=previous_state['last_epoch'] - 1
        )
        scheduler.load_state_dict(previous_state)
    else:
        # Create new scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )
    
    return scheduler


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
        model, training_state = _resume_from_checkpoint(args, device)
    else:
        model, training_state = _initialize_model_from_scratch(args, model_config, mask_time_prob, device)
    
    optimizer = _create_optimizer(model, lr, betas, weight_decay)
    
    if args.resume_pretraining and training_state['is_same_iteration']:
        optimizer.load_state_dict(training_state['optimizer_state'])
        optimizer.param_groups[0]['weight_decay'] = max(MIN_WEIGHT_DECAY, 
                                                         optimizer.param_groups[0]['weight_decay'])
        _ensure_min_dropout(model, DROPOUT_RESET_VALUE)
    
    lr_scheduler = _create_lr_scheduler(
        optimizer, 
        args.training_steps, 
        WARMUP_RATIO,
        training_state['global_step'],
        training_state.get('lr_scheduler_state')
    )

    hubert = model
    global_step = training_state['global_step']
    best_val_loss = training_state['best_val_loss']
    best_val_accuracy = training_state['best_val_accuracy']
    patience_count = training_state['patience_count']

    start_epoch = global_step // len(train_loader)

    for epoch in range(start_epoch, epochs):
        hubert.train()
        logger.info(f"Epoch {epoch+1}/{epochs}")

        train_losses = []
        
        for ecg, attention_mask, ensemble_labels in tqdm(train_loader, total=len(train_loader)):
            global_step += 1
            
            # Move data to device
            ecg = ecg.to(device) 
            attention_mask = attention_mask.to(device)
            ensemble_labels = ensemble_labels.to(device)
            
            with torch.amp.autocast('cuda'):
                # Forward pass
                encoder_output = hubert(
                    ecg, 
                    attention_mask=attention_mask, 
                    output_attentions=False, 
                    output_hidden_states=False, 
                    return_dict=True
                )
                
                mask = encoder_output['mask_time_indices']
                ensemble_logits = hubert.logits(encoder_output['last_hidden_state'])

                # Compute ensemble loss
                ensemble_labels = ensemble_labels.transpose(0, 1)
                
                # Vectorized loss computation
                masked_losses = []
                unmasked_losses = []

                for labels, logits in zip(ensemble_labels, ensemble_logits):
                    # labels: (batch_size, seq_len), logits: (batch_size, seq_len, vocab_size)
                    masked_losses.append(F.cross_entropy(logits[mask], labels[mask]))
                    unmasked_losses.append(F.cross_entropy(logits[~mask], labels[~mask]))
                
                masked_loss = sum(masked_losses)
                unmasked_loss = sum(unmasked_losses)
                
                loss = args.alpha * masked_loss + (1 - args.alpha) * unmasked_loss
                loss = loss / accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            train_losses.append(loss.item())

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

                    if global_step == 61:
                        torch.save(checkpoint, checkpoint_path / checkpoint_name)
                    
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
