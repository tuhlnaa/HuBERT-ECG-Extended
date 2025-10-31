import json
import logging
from pathlib import Path
from typing import Any, Dict
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
DROPOUT_ADJUSTMENT = 0.05
DROPOUT_RESET_VALUE = 0.1
WEIGHT_DECAY_MULTIPLIER = 5.0


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    patience: int
    lr: float
    betas: tuple[float, float]
    weight_decay: float
    accumulation_steps: int
    mask_time_prob: float


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


def _ensure_min_dropout(model, min_dropout):
    """Ensure all dropout modules have at least the minimum dropout rate."""
    for name, module in model.named_modules():
        if 'dropout' in name:
            module.p = max(min_dropout, module.p)


def resume_from_checkpoint(args, device):
    """Load model and training state from checkpoint."""
    if args.load_path:
        model_path = args.load_path
    elif args.pretrained_path:
        model_path = args.pretrained_path

    checkpoint_name = model_path.split('/')[-1]
    logger.info(f"Loading checkpoint {checkpoint_name} to resume pretraining")
    
    checkpoint = torch.load(model_path, map_location = 'cpu', weights_only=False)
    
    # Validate checkpoint
    assert checkpoint['pretraining_vocab_sizes'] == args.vocab_sizes, \
        "Vocab sizes mismatch between checkpoint and args"

    # Create config
    config = checkpoint['model_config']

    # Initialize and load model
    model = HuBERT(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Handle iteration switch
    if args.pretrained_path:
        logger.info("Switching to another pretraining iteration: "
                   "reinitializing label embedding and restoring dropouts...")
        # Reinitialize label embeddings for new training iteration
        model.label_embedding = nn.ModuleList([
            nn.Embedding(vocab_size, model.config.classifier_proj_size)
            for vocab_size in args.vocab_sizes
        ])
        # Reset dropout rates for encoder layers
        for name, module in model.named_modules():
            if 'dropout' in name and 'encoder.layers' in name:
                module.p = DROPOUT_RESET_VALUE
    
    # Prepare training state
    training_state = {
        'global_step': checkpoint['global_step'],
        'best_val_loss': checkpoint['best_val_loss'],
        'patience_count': checkpoint['patience_count'],
        'best_val_accuracy': checkpoint['best_val_accuracy'],
        'optimizer_state': checkpoint['optimizer_state_dict'],
        'scheduler_state': checkpoint['scheduler_state_dict']
    }

    return model, training_state


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


def _load_json_config(filename: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    config_path = Path(__file__).parents[1] / "configs" / filename
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)
    

def initialize_model_from_scratch(args, mask_time_prob, device):
    """Initialize model from scratch with given configuration."""
    logger.info("Building a model from zero to start training...")

    # Get model configuration from JSON
    configs = _load_json_config("model_configs.json")
    model_config = configs[args.largeness]

    # Get conv configuration from JSON
    configs = _load_json_config("conv_configs.json")
    key = "null" if args.downsampling_factor is None else str(args.downsampling_factor)
    conv_configs = configs[key]
    
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
        'optimizer_state': None,
        'scheduler_state': None
    }
    
    logger.info("Model built.")
    return model, training_state

