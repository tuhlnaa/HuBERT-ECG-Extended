"""
Test script for ECG dataset loading and validation.

Uusage:
python test/usage_dataset.py ./reproducibility/ptb/ptb_train_0.csv ./reproducibility/ptb/ptb_test_0.csv 8 --random_crop
python test/usage_dataset.py ./reproducibility/ptb/ptb_train_0.csv ./reproducibility/ptb/ptb_test_0.csv 8 --random_crop --downsample_factor 5

"""

import argparse
import sys
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table
from rich import box

# Import custom modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from HuBert_ECG.config import RichPrinter, init_seeds
from HuBert_ECG.dataset import ECGDataset

console = Console()


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
    
    return DataLoader(
        dataset,
        collate_fn=dataset.collate,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True,
    )


def print_batch_info(
    batch_idx: int,
    ecg_data: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    """Print information about a single batch using rich tables."""
    table = Table(title=f"Batch {batch_idx + 1}", box=box.ROUNDED)
    
    table.add_column("Tensor", style="cyan", no_wrap=True)
    table.add_column("Shape", style="bright_white")
    table.add_column("Dtype", style="green")
    table.add_column("Device", style="yellow")
    table.add_column("Range", style="bright_cyan")
    
    table.add_row(
        "ECG Data",
        str(ecg_data.shape),
        str(ecg_data.dtype),
        str(ecg_data.device),
        f"[{ecg_data.min():.4f}, {ecg_data.max():.4f}]",
    )
    
    table.add_row(
        "Attention Mask",
        str(attention_mask.shape),
        str(attention_mask.dtype),
        str(attention_mask.device),
        f"[{attention_mask.min()}, {attention_mask.max()}]",
    )
    
    table.add_row(
        "Labels",
        str(labels.shape),
        str(labels.dtype),
        str(labels.device),
        f"[{labels.min():.4f}, {labels.max():.4f}]",
    )
    
    console.print(table)
    console.print()


def test_dataloaders(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_batches: int = 2,
) -> None:
    """Test dataloaders by iterating through batches."""
    
    # Dataset Statistics
    stats_table = Table(title="Dataset Statistics", box=box.DOUBLE)
    stats_table.add_column("Metric", style="cyan", no_wrap=True)
    stats_table.add_column("Training", style="green", justify="right")
    stats_table.add_column("Validation", style="yellow", justify="right")
    
    stats_table.add_row("Samples", str(len(train_loader.dataset)), str(len(val_loader.dataset)))
    stats_table.add_row("Batches", str(len(train_loader)), str(len(val_loader)))
    
    console.print(stats_table)
    console.print()
    
    for batch_idx, (ecg_data, attention_mask, labels) in enumerate(train_loader):
        print_batch_info(batch_idx, ecg_data, attention_mask, labels)
        if batch_idx >= num_batches - 1:
            break
    
    for batch_idx, (ecg_data, attention_mask, labels) in enumerate(val_loader):
        print_batch_info(batch_idx, ecg_data, attention_mask, labels)
        if batch_idx >= num_batches - 1:
            break


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test ECG dataset loading and preprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument("train_csv", type=str, help="Path to training dataset CSV file",)
    required.add_argument("val_csv", type=str, help="Path to validation dataset CSV file",)
    required.add_argument("batch_size",type=int, help="Batch size for DataLoader",)

    # Optional arguments
    parser.add_argument("--ecg_dir", type=str, default="./output/PTB", help="Directory containing ECG data files",)
    parser.add_argument("--downsample_factor", type=int, default=None, help="Downsampling factor for ECG signals",)
    parser.add_argument("--random_crop", action="store_true", help="Apply random 5-second crop augmentation")
    parser.add_argument("--label_start_idx", type=int, default=3, help="Starting column index for labels in CSV")
    parser.add_argument("--num_batches", type=int, default=1, help="Number of batches to display for testing")

    args = parser.parse_args()
    RichPrinter.print_config(args, "Test Configuration")
    
    return args


def main() -> None:
    """Main execution function."""
    args = parse_args()
    init_seeds()
    
    # Create dataloaders
    train_loader = create_dataloader(
        csv_path=args.train_csv,
        ecg_dir=args.ecg_dir,
        batch_size=args.batch_size,
        label_start_idx=args.label_start_idx,
        downsample_factor=args.downsample_factor,
        random_crop=args.random_crop,
        shuffle=True,
    )
    
    val_loader = create_dataloader(
        csv_path=args.val_csv,
        ecg_dir=args.ecg_dir,
        batch_size=args.batch_size,
        label_start_idx=args.label_start_idx,
        downsample_factor=args.downsample_factor,
        random_crop=args.random_crop,
        shuffle=False,
    )
    
    # Test dataloaders
    test_dataloaders(train_loader, val_loader, args.num_batches)


if __name__ == "__main__":
    main()

"""
        Dataset Statistics
╔═════════╦══════════╦════════════╗
║ Metric  ║ Training ║ Validation ║
╠═════════╬══════════╬════════════╣
║ Samples ║      406 ║        107 ║
║ Batches ║       50 ║         13 ║
╚═════════╩══════════╩════════════╝

                                        Batch 1
╭────────────────┬────────────────────────┬───────────────┬────────┬───────────────────╮
│ Tensor         │ Shape                  │ Dtype         │ Device │ Range             │
├────────────────┼────────────────────────┼───────────────┼────────┼───────────────────┤
│ ECG Data       │ torch.Size([8, 30000]) │ torch.float32 │ cpu    │ [-1.0000, 1.0000] │
│ Attention Mask │ torch.Size([8, 30000]) │ torch.int64   │ cpu    │ [1, 1]            │
│ Labels         │ torch.Size([8, 14])    │ torch.float32 │ cpu    │ [0.0000, 1.0000]  │
╰────────────────┴────────────────────────┴───────────────┴────────┴───────────────────╯
"""