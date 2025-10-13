"""
Configuration management for PyTorch training using OmegaConf.
Based on: https://github.com/huggingface/pytorch-image-models/blob/main/train.py
"""
import argparse
import json
import torch
import random
import logging
import numpy as np

from torch.backends import cudnn
from typing import Any, Dict, Union
from pathlib import Path
from rich.table import Table
from rich import box, print
from rich.pretty import Pretty
from rich.logging import RichHandler
from omegaconf import OmegaConf


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", 
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


def init_seeds(seed: int = 42, cuda_deterministic: bool = True) -> None:
    """Initialize random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        cuda_deterministic: If True, use deterministic CUDA operations (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if cuda_deterministic:
            cudnn.deterministic = True
            cudnn.benchmark = False
        else:
            cudnn.deterministic = False
            cudnn.benchmark = True


class ConfigurationManager:
    """Handles configuration loading and validation."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load and validate configuration from JSON file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with config_file.open('r') as f:
            config = json.load(f)

        return config


class RichPrinter:
    @staticmethod
    def print_dict(data: Union[Dict, str], title: str = "Dictionary") -> None:
        """Print dictionary details in a structured table.
        
        Args:
            data: Dictionary object or JSON string to display
            title: Title for the table display
        """
        table = Table(title=title, box=box.ROUNDED)
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        # Handle JSON string input
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                print(f"[red]Error parsing JSON string: {e}[/red]")
                return
        
        # Ensure we have a dictionary
        if not isinstance(data, dict):
            print(f"[red]Error: Expected dictionary or JSON string, got {type(data)}[/red]")
            return
        
        # Add rows recursively for nested dictionaries
        def add_dict_to_table(d: Dict, prefix: str = "") -> None:
            for key, value in d.items():
                param_name = f"{prefix}{key}"
                if isinstance(value, dict):
                    add_dict_to_table(value, f"{param_name}.")
                else:
                    pretty_value = Pretty(value, indent_guides=False)
                    table.add_row(param_name, pretty_value)
        
        add_dict_to_table(data)
        print(table)
        print()  # Add spacing after table


    @staticmethod
    def print_config(config: Any, title: str = "Configuration") -> None:
        """Print configuration details in a structured table."""
        table = Table(title=title, box=box.ROUNDED)
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        # Check if the config is an OmegaConf object
        if OmegaConf.is_config(config):
            # Convert OmegaConf to a dictionary
            config_dict = OmegaConf.to_container(config, resolve=True)
            
            # Add rows recursively for nested config
            def add_dict_to_table(d, prefix=""):
                for key, value in d.items():
                    param_name = f"{prefix}{key}"
                    if isinstance(value, dict):
                        add_dict_to_table(value, f"{param_name}.")
                    else:
                        pretty_value = Pretty(value, indent_guides=False)
                        table.add_row(param_name, pretty_value)
            
            add_dict_to_table(config_dict)
        else:
            # Handle argparse or other config types
            for key, value in vars(config).items():
                pretty_value = Pretty(value, indent_guides=False)
                table.add_row(key, pretty_value)
        
        print(table)
        print()  # Add spacing after table


def create_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="Train Hubert-ECG")
    
    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument("train_iteration", type=int, choices=[1, 2, 3], help="Hubert training iteration in {1, 2, 3}")
    required.add_argument("path_to_dataset_csv_train", type=str, help="Path to the csv file containing the training dataset")
    required.add_argument("path_to_dataset_csv_val", type=str, help="Path to the csv file containing the validation dataset")
    required.add_argument("ecg_dir", type=str, help="Directory containing ECG data files",)
    required.add_argument("vocab_size", type=int, help="Vocabulary size, i.e. num of labels/clusters")
    required.add_argument("patience", type=int, help="Patience for early stopping")
    required.add_argument("batch_size", type=int, help="Batch size")
    required.add_argument("target_metric", type=str, choices=["f1_score", "recall", "precision", "specificity", "auroc", "auprc", "accuracy"],
        help="Target metric (macro) to optimize during finetuning"
    )
    
    # Training schedule (mutually exclusive)
    schedule = parser.add_mutually_exclusive_group(required=True)
    schedule.add_argument("--training_steps", type=int, help="Number of training steps to perform")
    schedule.add_argument("--epochs", type=int, help="Number of epochs to perform")
    
    # Model initialization (mutually exclusive)
    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument("--resume_finetuning", action="store_true", help="Whether to resume finetuning")
    init_group.add_argument("--random_init", action="store_true", help="Whether to initialize the model with random weights")
    
    # General optional arguments
    parser.add_argument('--experiment_name', type=str, default="finetune", help='Name for the experiment')
    parser.add_argument('--output_dir', type=str, default="./output", help='Path to save model and results')
    parser.add_argument("--sweep_dir", type=str, default=".", help="Sweep directory. Default `.`")
    parser.add_argument("--ramp_up_perc", type=float, default=0.08, help="Percentage of training steps for the ramp up phase. Default 0.08")
    parser.add_argument("--val_interval", type=int, help="Training steps to wait before validation. Required if training_steps is used")
    parser.add_argument("--downsampling_factor", type=int, help="Downsampling factor to apply to the ECG signal")
    parser.add_argument("--random_crop", action="store_true", help="Whether to perform random crop of 5 sec as data augmentation")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Number of batch gradients to accumulate before updating model params")
    parser.add_argument("--label_start_index", type=int, default=3, help="Index of the first label in the dataset csv file")
    
    # Model architecture
    parser.add_argument("--load_path", type=str, help="Path to a model checkpoint to load for starting/resuming fine-tuning")
    parser.add_argument("--largeness", type=str, choices=["small", "base", "large"], help="Model largeness in case of random initialization")
    parser.add_argument("--classifier_hidden_size", type=int, help="Hidden size of the MLP head. If None, uses linear classifier")
    parser.add_argument("--use_label_embedding", action="store_true", help="Whether to use label embeddings in the classification head")

    # Training strategy
    parser.add_argument("--freezing_steps", type=int, help="Number of finetuning steps to keep frozen the base model weights")
    parser.add_argument("--unfreeze_conv_embedder", action="store_true",help="Whether to unfreeze the convolutional feature extractor during fine-tuning")
    parser.add_argument("--transformer_blocks_to_unfreeze", type=int, default=0, help="Number of transformer blocks to unfreeze after freezing_steps")
    
    # Optimization
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--layer_wise_lr", action="store_true", help="Whether to use layer-wise learning rate")
    parser.add_argument("--weight_decay_mult", type=float, default=1.0, help="Weight decay multiplier. Default 1.0 (i.e. WD=0.01)")
    parser.add_argument("--model_dropout_mult", type=float, default=0.0, help="Model dropout multiplier. Default 0.0 (i.e. dropout=0.1)")
    parser.add_argument("--finetuning_layerdrop", type=float, default=0.1, help="Layerdrop for the finetuning phase")
    
    # Regularization
    parser.add_argument("--dynamic_reg", action="store_true", help="Whether to apply dynamic regularization to the model")
    parser.add_argument("--intervals_for_penalty", type=int, default=3, help="Number of validation intervals with worsening performance before applying regularization")
    parser.add_argument("--use_loss_weights", action="store_true", help="Whether to use loss weights in the loss function")
    
    # Task configuration
    parser.add_argument('--task', type=str, choices=["multi_class", "multi_label", "regression"], default="multi_label", help="Task to perform")
    
    # Logging
    parser.add_argument("--wandb_run_name", type=str, help="The name to give to this run")

    args = parser.parse_args()

    # Print configuration
    RichPrinter.print_config(args, "Configuration")

    # Validate arguments
    validate_args(args)

    return args


def validate_args(args):
    """Validate argument combinations and constraints."""
    errors = []
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available. CPU finetuning not supported")
        exit(1)
    
    # Validate ramp_up_perc range
    if not 0 <= args.ramp_up_perc <= 1:
        errors.append("ramp_up_perc must be in [0, 1] range")
    
    # Validate val_interval requirement
    if args.training_steps is not None and args.val_interval is None:
        errors.append("val_interval must be provided when using training_steps")
    
    # Validate divisibility constraints
    if args.training_steps and args.val_interval:
        if args.training_steps % args.val_interval != 0:
            errors.append(f"training_steps ({args.training_steps}) must be divisible by val_interval ({args.val_interval})")
    
    if args.training_steps and args.accumulation_steps > 1:
        if args.training_steps % args.accumulation_steps != 0:
            errors.append(f"training_steps ({args.training_steps}) must be divisible by accumulation_steps ({args.accumulation_steps})")
    
    # Validate load_path requirement
    if not args.random_init and args.load_path is None:
        errors.append("load_path must be provided when not using random_init")
    
    # Validate freezing_steps
    if args.freezing_steps is not None and args.training_steps is not None:
        if args.freezing_steps > args.training_steps:
            errors.append(f"freezing_steps ({args.freezing_steps}) cannot be greater than training_steps ({args.training_steps})")
    
    # Validate random_init requirements
    if args.random_init and args.largeness is None:
        errors.append("largeness must be provided when using random_init")
    
    # Validate dynamic_reg requirements
    if args.dynamic_reg and args.patience < args.intervals_for_penalty:
        errors.append(f"patience ({args.patience}) must be >= intervals_for_penalty ({args.intervals_for_penalty}) when using dynamic_reg")
    
    # Warnings
    if args.random_init and args.load_path is not None:
        logger.warning("random_init is provided. load_path will be ignored")
    
    # Raise all errors at once
    if errors:
        raise ValueError("Argument validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
