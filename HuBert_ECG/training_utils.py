import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim

from math import ceil
from pathlib import Path
from rich.logging import RichHandler
from transformers import HubertConfig, get_linear_schedule_with_warmup
from typing import Any, Dict, Optional
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
WARMUP_RATIO = 0.08
DROPOUT_RESET_VALUE = 0.1
MIN_WEIGHT_DECAY = 0.01


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


def _load_json_config(filename: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    config_path = Path(__file__).parents[1] / "configs" / filename
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def _ensure_min_dropout(model: nn.Module, min_dropout: float) -> None:
    """Ensure all dropout modules have at least the minimum dropout rate."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = max(min_dropout, module.p)


def _create_optimizer(model: nn.Module, learning_rate: float, weight_decay_mult: float) -> optim.AdamW:
    """Create AdamW optimizer with specified parameters."""
    weight_decay = max(0.0, 0.01 * weight_decay_mult)
    return optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1E-09,
        weight_decay=weight_decay
    )


def _create_scheduler(
    optimizer: optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = WARMUP_RATIO,
    current_step: int = 0,
    previous_state: Optional[Dict] = None
) -> Any:
    """Create learning rate scheduler with warmup and optional state restoration."""
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


def resume_from_checkpoint(args, device):
    """Load model and training state from checkpoint."""
    if args.load_path:
        model_path = args.load_path
    elif args.pretrained_path:
        model_path = args.pretrained_path

    logger.info(f"Loading checkpoint {Path(model_path).name}...")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Validate checkpoint
    assert checkpoint['pretraining_vocab_sizes'] == args.vocab_sizes, \
        "Vocab sizes mismatch between checkpoint and args"

    # Initialize and load model
    model = HuBERT(checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Handle iteration switch (using pretrained model for new iteration)
    if args.pretrained_path:
        logger.info(
            "Switching to another pretraining iteration: "
            "reinitializing label embedding and restoring dropouts..."
        )
        # Reinitialize label embeddings for new training iteration
        model.label_embedding = nn.ModuleList([
            nn.Embedding(vocab_size, model.config.classifier_proj_size)
            for vocab_size in args.vocab_sizes
        ])
        # Reset dropout rates for encoder layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout) and 'encoder.layers' in name:
                module.p = DROPOUT_RESET_VALUE
    
    # Create and restore optimizer
    optimizer = _create_optimizer(model, args.lr, args.weight_decay_mult)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Ensure minimum weight decay
    for param_group in optimizer.param_groups:
        param_group['weight_decay'] = max(MIN_WEIGHT_DECAY, param_group['weight_decay'])
    
    # Ensure minimum dropout
    _ensure_min_dropout(model, DROPOUT_RESET_VALUE)

    scheduler = _create_scheduler(
        optimizer, 
        args.training_steps, 
        WARMUP_RATIO,
        checkpoint['global_step'],
        checkpoint['scheduler_state_dict']
    )

    # Prepare training state
    training_state = {
        'global_step': checkpoint['global_step'],
        'best_val_loss': checkpoint['best_val_loss'],
        'patience_count': checkpoint['patience_count'],
        'best_val_accuracy': checkpoint['best_val_accuracy'],
    }

    return model, optimizer, scheduler, training_state
    

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

    optimizer = _create_optimizer(model, args.lr, args.weight_decay_mult)
    scheduler = _create_scheduler(optimizer, args.training_steps)

    training_state = {
        'global_step': 0,
        'best_val_loss': float('inf'),
        'best_val_accuracy': 0.0,
        'patience_count': 0,
    }
    
    return model, optimizer, scheduler, training_state
