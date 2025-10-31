import logging
import torch

import torch.nn as nn
import torch.optim as optim

from dataclasses import dataclass
from math import ceil
from rich.logging import RichHandler
from transformers import HubertConfig, get_linear_schedule_with_warmup
# from transformers.models.hubert.modeling_hubert import compute_mask_indices

# Import custom modules
from hubert_ecg import HuBERTECG as HuBERT, HuBERTECGConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", 
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

# Constants
EPS = 1E-09
DROPOUT_ADJUSTMENT = 0.05
DROPOUT_RESET_VALUE = 0.1
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


def _resume_from_checkpoint(args, device):
    """Load model and training state from checkpoint."""
    checkpoint_name = args.load_path.split('/')[-1]
    logger.info(f"Loading checkpoint {checkpoint_name} to resume pretraining")
    
    checkpoint = torch.load(args.load_path, map_location=torch.device('cpu'), weights_only=False)
    
    # Validate checkpoint
    assert checkpoint['pretraining_vocab_sizes'] == args.vocab_sizes, \
        "Vocab sizes mismatch between checkpoint and args"

    # Create config
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
