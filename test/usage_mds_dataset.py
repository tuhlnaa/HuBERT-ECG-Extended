"""
Test script for MDS dataset loading and validation.

Usage:
python test/usage_mds_dataset.py ./output/mds_train ./output/mds_test 16
python test/usage_mds_dataset.py ./output/mds_train ./output/mds_test 16 --downsample_factor 5
python test/usage_mds_dataset.py ./output/mds_train ./output/mds_test 16 --year_filter 2020

"""

import argparse
import sys
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table
from rich import box
from streaming import StreamingDataset

# Import custom modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from HuBert_ECG.config import RichPrinter, init_seeds

console = Console()


class ECGMDSDataset(StreamingDataset):
    """
    PyTorch Dataset wrapper for MDS format ECG data.
    
    Args:
        mds_dir: Directory containing MDS files
        batch_size: Per-device batch size (required by StreamingDataset)
        downsample_factor: Optional downsampling factor
        year_filter: Optional year filter
    """
    
    def __init__(
        self,
        mds_dir: str,
        batch_size: int,
        downsample_factor: int = None,
        year_filter: int = None,
        **kwargs
    ):
        super().__init__(local=mds_dir, batch_size=batch_size, **kwargs)
        self.downsample_factor = downsample_factor
        self.year_filter = year_filter
        
    def __getitem__(self, idx: int):
        """Get a single sample from the dataset."""
        sample = super().__getitem__(idx)
        
        # Filter by year if specified
        if self.year_filter is not None and sample.get('year') != self.year_filter:
            return None
        
        # Extract ECG data
        ecg_data = sample['ecg_data']  # Shape: [12, 5000]
        
        # Apply downsampling if specified
        if self.downsample_factor is not None and self.downsample_factor > 1:
            ecg_data = ecg_data[:, ::self.downsample_factor]
        
        # Convert to tensor
        ecg_tensor = torch.from_numpy(ecg_data).float()
        
        # Create attention mask (all ones for valid data)
        attention_mask = torch.ones(ecg_tensor.shape[1], dtype=torch.long)
        
        # Create dummy labels (modify based on your actual label structure)
        # For now, using zeros as placeholder
        labels = torch.zeros(1, dtype=torch.float32)
        
        return ecg_tensor, attention_mask, labels


def create_mds_dataloader(
    mds_dir: str,
    batch_size: int,
    downsample_factor: int = None,
    year_filter: int = None,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for MDS format ECG data.
    
    Args:
        mds_dir: Directory containing MDS files
        batch_size: Batch size for DataLoader (also passed to StreamingDataset)
        downsample_factor: Optional downsampling factor
        year_filter: Optional year filter
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    dataset = ECGMDSDataset(
        mds_dir=mds_dir,
        batch_size=batch_size,
        downsample_factor=downsample_factor,
        year_filter=year_filter,
        shuffle=shuffle,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda batch: tuple(torch.stack(x) for x in zip(*[b for b in batch if b is not None])),
    )
    
    return dataloader


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


def print_mds_metadata(mds_dir: str) -> None:
    """Print MDS dataset metadata if available."""
    import json
    
    metadata_file = Path(mds_dir) / "conversion_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        meta_table = Table(title="MDS Dataset Metadata", box=box.DOUBLE)
        meta_table.add_column("Property", style="cyan", no_wrap=True)
        meta_table.add_column("Value", style="bright_white")
        
        for key, value in metadata.items():
            if isinstance(value, list):
                value = str(value)
            meta_table.add_row(str(key), str(value))
        
        console.print(meta_table)
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
    
    console.print("[bold green]Training Batches:[/bold green]")
    for batch_idx, (ecg_data, attention_mask, labels) in enumerate(train_loader):
        print_batch_info(batch_idx, ecg_data, attention_mask, labels)
        if batch_idx >= num_batches - 1:
            break
    
    console.print("[bold yellow]Validation Batches:[/bold yellow]")
    for batch_idx, (ecg_data, attention_mask, labels) in enumerate(val_loader):
        print_batch_info(batch_idx, ecg_data, attention_mask, labels)
        if batch_idx >= num_batches - 1:
            break


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test MDS ECG dataset loading and preprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument("train_mds_dir", type=str, help="Path to training MDS directory")
    required.add_argument("val_mds_dir", type=str, help="Path to validation MDS directory")
    required.add_argument("batch_size", type=int, help="Batch size for DataLoader")

    # Optional arguments
    parser.add_argument("--downsample_factor", type=int, default=None, help="Downsampling factor for ECG signals")
    parser.add_argument("--year_filter", type=int, default=None, help="Filter data by year (e.g., 2020)")
    parser.add_argument("--num_batches", type=int, default=1, help="Number of batches to display for testing")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader worker processes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    RichPrinter.print_config(args, "Test Configuration")
    
    return args


def main():
    """Main function to test MDS dataset loading."""
    args = parse_args()
    
    # Initialize random seeds
    init_seeds(args.seed)
    
    # Print metadata for both datasets
    console.print("[bold cyan]Training Dataset Metadata:[/bold cyan]")
    print_mds_metadata(args.train_mds_dir)
    
    console.print("[bold cyan]Validation Dataset Metadata:[/bold cyan]")
    print_mds_metadata(args.val_mds_dir)
    
    # Create dataloaders
    console.print("[bold]Creating DataLoaders...[/bold]")
    
    train_loader = create_mds_dataloader(
        mds_dir=args.train_mds_dir,
        batch_size=args.batch_size,
        downsample_factor=args.downsample_factor,
        year_filter=args.year_filter,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = create_mds_dataloader(
        mds_dir=args.val_mds_dir,
        batch_size=args.batch_size,
        downsample_factor=args.downsample_factor,
        year_filter=args.year_filter,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    # Test dataloaders
    console.print("[bold]Testing DataLoaders...[/bold]\n")
    test_dataloaders(train_loader, val_loader, num_batches=args.num_batches)
    
    console.print("[bold green]✓ Dataset loading test completed successfully![/bold green]")


if __name__ == "__main__":
    main()
"""
Training Dataset Metadata:
                                      MDS Dataset Metadata
╔════════════════╦═════════════════════════════════════════════════════════════════════════════╗
║ Property       ║ Value                                                                       ║
╠════════════════╬═════════════════════════════════════════════════════════════════════════════╣
║ total_samples  ║ 45239                                                                       ║
║ year_filter    ║ None                                                                        ║
║ channel_labels ║ ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] ║
║ ecg_shape      ║ [12, 5000]                                                                  ║
║ dtype          ║ float32                                                                     ║
║ compression    ║ zstd:3                                                                      ║
║ shard_size_mb  ║ 64                                                                          ║
╚════════════════╩═════════════════════════════════════════════════════════════════════════════╝

╭────────────────┬────────────────────────────┬───────────────┬────────┬───────────────────╮
│ Tensor         │ Shape                      │ Dtype         │ Device │ Range             │
├────────────────┼────────────────────────────┼───────────────┼────────┼───────────────────┤
│ ECG Data       │ torch.Size([16, 12, 5000]) │ torch.float32 │ cpu    │ [-4.2110, 3.0840] │
│ Attention Mask │ torch.Size([16, 5000])     │ torch.int64   │ cpu    │ [1, 1]            │
│ Labels         │ torch.Size([16, 1])        │ torch.float32 │ cpu    │ [0.0000, 0.0000]  │
╰────────────────┴────────────────────────────┴───────────────┴────────┴───────────────────╯
"""