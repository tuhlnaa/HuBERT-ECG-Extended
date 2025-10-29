"""
Configuration management for PyTorch training using OmegaConf.
Based on: https://github.com/huggingface/pytorch-image-models/blob/main/train.py
"""
import argparse
import json
import logging
import os
import random
import torch
import numpy as np

from omegaconf import OmegaConf
from pathlib import Path
from rich.logging import RichHandler
from rich.pretty import Pretty
from rich.table import Table
from torch.backends import cudnn
from rich import box, print
from typing import Any, Dict, Union

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


def create_finetuning_parser():
    """Create and configure argument parser for finetuning."""
    parser = argparse.ArgumentParser(description="Fine-tune Hubert-ECG")
    
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
    validate_finetuning_args(args)

    return args


def validate_finetuning_args(args):
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


def create_dumping_parser():
    """Create and configure argument parser for feature dumping."""
    parser = argparse.ArgumentParser(description="Dump features for Hubert-ECG training")
    
    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument("train_iteration", type=int, choices=[1, 2, 3], 
                          help="Training iteration to consider for dumping features. If 1, dump morphological features. If 2+, dump Hubert hidden features")
    required.add_argument("dataframe_path", type=str, help="Path to the dataframe object in csv format")
    required.add_argument("in_dir", type=str, help="Input directory where real files (those pointed by dataframe object) are")
    required.add_argument("dest_dir", type=str, help="Directory where to dump extracted features")
    
    # Percentage range arguments
    parser.add_argument("start_perc",type=float, default=0.0, help="Min percentage of the dataframe to dump features from. Used only when train_iteration = 1")
    parser.add_argument("end_perc", type=float, default=1.0, help="Max percentage of the dataframe to dump features from. Used only when train_iteration = 1")
    
    # Feature type selection (mutually exclusive)
    feature_type = parser.add_mutually_exclusive_group()
    feature_type.add_argument("--mfcc_only", action="store_true", help="If True, dump only MFCC features and derivatives. Used only when train_iteration = 1")
    feature_type.add_argument("--time_freq", action="store_true", help="If True, dump only time and frequency features. Used only when train_iteration = 1")

    # Model and processing arguments
    parser.add_argument("--hubert_path", type=str, help="Path to the Hubert model to use for extracting latent features. Used only with train_iteration > 1")
    parser.add_argument("--samp_rate", type=int, help="Sampling rate of the ECG signal from which features are extracted. Used only when train_iteration = 1 and when MFCC features are computed")
    parser.add_argument("--batch_size", type=int,default=1, help="Batch size to use when dumping latent features. Used only when train_iteration > 1")
    parser.add_argument("--output_layer", type=int, help="Output layer of HuBERT encoder from which to take latent features. Used only when train_iteration > 1")
    parser.add_argument("--save_csv_for_dumped_features", action="store_true", help="Whether to save a csv file containing references to dumped features. Helpful when clustering is the next step")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files (default: skip existing files)")
    
    args = parser.parse_args()
    
    # Print configuration
    RichPrinter.print_config(args, "Configuration")

    # Validate arguments
    validate_dumping_args(args)

    return args


def validate_dumping_args(args):
    """Validate argument combinations and constraints."""
    errors = []
    
    # Validate train_iteration range
    if args.train_iteration < 1 or args.train_iteration > 3:
        errors.append(f"train_iteration must be 1, 2 or 3. Got {args.train_iteration}")
    
    # Validate percentage ranges
    if not (0 <= args.start_perc <= 1):
        errors.append(f"start_perc must be in [0, 1] range. Got {args.start_perc}")
    
    if not (0 <= args.end_perc <= 1):
        errors.append(f"end_perc must be in [0, 1] range. Got {args.end_perc}")
    
    # Validate mutually exclusive feature types
    if args.mfcc_only and args.time_freq:
        errors.append("mfcc_only and time_freq are mutually exclusive")
    
    # Validate hubert_path requirement for train_iteration > 1
    if args.train_iteration > 1 and args.hubert_path is None:
        errors.append("hubert_path must be specified when train_iteration is 2 or 3")
    
    # Validate output_layer requirement for train_iteration > 1
    if args.train_iteration > 1 and args.output_layer is None:
        errors.append("output_layer must be provided when train_iteration > 1")
    
    # Validate hubert_path file existence
    if args.train_iteration > 1 and args.hubert_path is not None:
        if not os.path.isfile(args.hubert_path):
            errors.append(f"hubert_path must be a valid file path. Got {args.hubert_path}")
    
    # Validate samp_rate requirement for MFCC features
    if args.train_iteration == 1:
        needs_mfcc = args.mfcc_only or (not args.mfcc_only and not args.time_freq)
        if needs_mfcc and args.samp_rate is None:
            errors.append("samp_rate must be provided when dumping features that include MFCC")
    
    # Warnings for unnecessary arguments
    if args.mfcc_only and args.train_iteration > 1:
        logger.warning("mfcc_only is not needed when train_iteration is 2 or 3. Ignoring it")
    
    if args.train_iteration == 1 and args.hubert_path is not None:
        logger.warning("hubert_path is not needed when train_iteration is 1. Ignoring it")
    
    if args.train_iteration == 1 and args.batch_size is not None:
        logger.warning("batch_size is not needed when train_iteration is 1. Ignoring it")
    
    if args.train_iteration == 1 and args.output_layer is not None:
        logger.warning("output_layer is not needed when train_iteration is 1. Ignoring it")
    
    if not args.mfcc_only and not args.time_freq and args.train_iteration == 1:
        logger.warning("Neither mfcc_only nor time_freq provided. Dumping mixed features")
    
    if args.time_freq and args.samp_rate is not None:
        logger.warning("samp_rate not necessary when dumping only time_freq features. Ignoring it")
    
    # Raise all errors at once
    if errors:
        raise ValueError("Argument validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


def create_clustering_parser():
    """Create and configure argument parser for clustering."""
    parser = argparse.ArgumentParser(description="Cluster ECG features or representations")
    
    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument("path_to_dataset_csv", type=str, help="Path to the dataset in csv format to use")
    required.add_argument("in_dir", type=str, help="Path to the directory containing the features to cluster")
    required.add_argument("train_iteration", type=int, help="Iteration of the training")
    required.add_argument("batch_size", type=int, help="Batch size")
    
    # Mode selection
    parser.add_argument("--cluster", action="store_true", help="Whether to cluster or evaluate a model")
    
    # Clustering arguments
    parser.add_argument("--n_clusters_start", type=int, help="Initial number of clusters. Required when --cluster is used")
    parser.add_argument("--n_clusters_end", type=int, help="Final number of clusters. Required when --cluster is used")
    parser.add_argument("--step", type=int, help="Step between two consecutive number of clusters. Required when --cluster is used")
    
    # Model and layer arguments
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model to evaluate or to load in order to resume clustering")
    parser.add_argument("--layer", type=int, default=None, help="In case train_iteration >= 2, which hidden layer latents were extracted from")
    
    args = parser.parse_args()
    
    # Print configuration
    RichPrinter.print_config(args, "Configuration")
    
    # Validate arguments
    validate_clustering_args(args)

    return args


def validate_clustering_args(args):
    """Validate argument combinations and constraints."""
    errors = []
    
    # Validate clustering mode requirements
    if args.cluster:
        if args.n_clusters_start is None:
            errors.append("n_clusters_start must be specified when --cluster is used")
        
        if args.n_clusters_end is None:
            errors.append("n_clusters_end must be specified when --cluster is used")
        
        if args.step is None:
            errors.append("step must be specified when --cluster is used")
        
        # Validate cluster range
        if args.n_clusters_start is not None and args.n_clusters_end is not None:
            if args.n_clusters_start > args.n_clusters_end:
                errors.append(f"n_clusters_start ({args.n_clusters_start}) must be <= n_clusters_end ({args.n_clusters_end})")
        
        # Validate step
        if args.step is not None and args.step <= 0:
            errors.append(f"step must be positive. Got {args.step}")
    
    # Validate evaluation mode requirements
    if not args.cluster:
        if args.model_path is None:
            errors.append("model_path must be specified when not in clustering mode")
    
    # Validate layer requirement for train_iteration >= 2
    if args.train_iteration >= 2 and args.layer is None:
        errors.append("layer must be specified when train_iteration >= 2")
    
    # Validate model_path file existence
    if args.model_path is not None and not os.path.isfile(args.model_path):
        errors.append(f"model_path must be a valid file path. Got {args.model_path}")
    
    # Warnings for unnecessary arguments
    if not args.cluster and args.n_clusters_start is not None:
        logger.warning("n_clusters_start is not needed in evaluation mode. Ignoring it")
    
    if not args.cluster and args.n_clusters_end is not None:
        logger.warning("n_clusters_end is not needed in evaluation mode. Ignoring it")
    
    if not args.cluster and args.step is not None:
        logger.warning("step is not needed in evaluation mode. Ignoring it")
    
    if args.train_iteration < 2 and args.layer is not None:
        logger.warning("layer is not needed when train_iteration < 2. Ignoring it")
    
    # Raise all errors at once
    if errors:
        raise ValueError("Argument validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


def create_training_parser():
    """Create and configure argument parser for training."""
    parser = argparse.ArgumentParser(description="Train Hubert-ECG")
    
    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument("train_iteration", type=int, choices=[1, 2, 3], help="Hubert training iteration in {1, 2, 3}")
    required.add_argument("path_to_dataset_csv_train", type=str, help="Path to the csv file containing the training dataset")
    required.add_argument("path_to_dataset_csv_val", type=str, help="Path to the csv file containing the validation dataset")
    required.add_argument("batch_size", type=int, help="Batch size")
    required.add_argument("patience", type=int, help="Patience for early stopping")

    # Pretraining-specific required arguments
    required.add_argument("mask_time_prob", type=float, help="Probability of masking a time step in the input sequence")
    required.add_argument("alpha", type=float, help="Alpha weight in the pretraining loss function")
    required.add_argument("kmeans_path", type=str, help="Path to a file that contains paths to KMeans models")
    required.add_argument("train_features_path", type=str, help="In case of pretraining or resumed pretraining, the path from which training features to cluster can be loaded")
    required.add_argument("val_features_path", type=str, help="In case of pretraining or resumed pretraining, the path from which validation features to cluster can be loaded")
    required.add_argument("vocab_sizes", type=int, nargs="+", help="Vocabulary sizes, i.e. num of labels/clusters per each task/clustering model")

    # Training schedule (mutually exclusive)
    schedule = parser.add_mutually_exclusive_group(required=True)
    schedule.add_argument("--training_steps", type=int, help="Number of training steps to perform")
    schedule.add_argument("--epochs", type=int, help="Number of epochs to perform")

    # Model initialization (mutually exclusive)
    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument("--resume_pretraining", action="store_true", help="Whether to resume pretraining")

    # General optional arguments
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Number of batch gradients to accumulate before updating model params")
    parser.add_argument("--val_interval", type=int, help="Training steps to wait before validation. Required if training_steps is used")
    parser.add_argument("--downsampling_factor", type=int, help="Downsampling factor to apply to the ECG signal")

    # Model architecture
    parser.add_argument("--load_path", type=str, help="Path to a model checkpoint to load for starting/resuming fine-tuning")
    parser.add_argument("--largeness", type=str, choices=["small", "base", "large"], help="Model largeness in case of random initialization")

    # Optimization
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay_mult", type=float, default=1.0, help="Weight decay multiplier. Default 1.0 (i.e. WD=0.01)")
    parser.add_argument("--model_dropout_mult", type=float, default=0.0, help="Model dropout multiplier. Default 0.0 (i.e. dropout=0.1)")

    # Regularization
    parser.add_argument("--dynamic_reg", action="store_true", help="Whether to apply dynamic regularization to the model")
    parser.add_argument("--intervals_for_penalty", type=int, default=4, help="Number of validation intervals with worsening performance before applying regularization")

    # Logging
    parser.add_argument("--wandb_run_name", type=str, help="The name to give to this run")

    args = parser.parse_args()

    # Print configuration
    RichPrinter.print_config(args, "Configuration")
    
    # Validate arguments
    validate_training_args(args)
    
    return args


def validate_training_args(args):
    """Validate argument combinations and constraints."""
    errors = []
    
    # Validate train_iteration range
    if args.train_iteration < 1 or args.train_iteration > 3:
        errors.append(f"train_iteration must be in [1, 3] range. Got {args.train_iteration}")
    
    # Validate mutually exclusive training schedule
    if args.epochs is None and args.training_steps is None:
        errors.append("Either epochs or training_steps must be provided")
    
    if args.epochs is not None and args.training_steps is not None:
        errors.append("epochs and training_steps cannot be provided at the same time")
    
    # Validate training_steps divisibility by val_interval
    if args.training_steps is not None and args.val_interval is not None:
        if args.training_steps % args.val_interval != 0:
            errors.append(f"training_steps must be divisible by val_interval. Got {args.training_steps} and {args.val_interval}")
    
    # Validate largeness choices
    if args.largeness is not None and args.largeness not in ["base", "large", "small"]:
        errors.append(f"largeness must be in [base, large, small]. Got {args.largeness}")
    
    # Validate mask_time_prob range
    if args.mask_time_prob <= 0.0 or args.mask_time_prob >= 1.0:
        errors.append(f"mask_time_prob must be in (0.0, 1.0) range. Got {args.mask_time_prob}")
    
    # Validate alpha range
    if args.alpha < 0.0 or args.alpha > 1.0:
        errors.append(f"alpha must be in [0.0, 1.0] range. Got {args.alpha}")
    
    # Validate file paths existence
    if not os.path.exists(args.kmeans_path):
        errors.append(f"kmeans_path must be a valid path. Got {args.kmeans_path}")
    
    if not os.path.exists(args.train_features_path):
        errors.append(f"train_features_path must be a valid path. Got {args.train_features_path}")
    
    if not os.path.exists(args.val_features_path):
        errors.append(f"val_features_path must be a valid path. Got {args.val_features_path}")
    
    # Validate resume_pretraining requires load_path
    if args.resume_pretraining and args.load_path is None:
        errors.append("load_path must be provided when resume_pretraining is specified")
    
    # Validate training_steps divisibility by accumulation_steps
    if args.accumulation_steps is not None and args.training_steps is not None:
        if args.training_steps % args.accumulation_steps != 0:
            errors.append(f"training_steps must be divisible by accumulation_steps. Got {args.training_steps} and {args.accumulation_steps}")
    
    # Warnings for potential issues
    if args.training_steps is not None and args.val_interval is None:
        logger.warning("val_interval not provided when using training_steps. Validation may not occur")
    
    # Raise all errors at once
    if errors:
        raise ValueError("Argument validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
