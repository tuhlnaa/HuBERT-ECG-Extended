"""
Convert ECG .npy files to MDS (Mosaic Data Shard) format.

This script converts ECG data stored in NumPy .npy files to the MDS format
for efficient data loading and streaming. Supports filtering by year based
on the filename date format.
"""

import argparse
import json
import logging
import sys
import numpy as np

from pathlib import Path
from rich.logging import RichHandler
from streaming import MDSWriter
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
from typing import Optional, Dict, Any

# Import custom modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from HuBert_ECG.config import RichPrinter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", 
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


def parse_filename(filename: str) -> Dict[str, Any]:
    """
    Parse ECG filename to extract metadata.
    Expected format: MUSE_YYYYMMDD_HHMMSS_*_tpi_*_*.npy
    """
    parts = filename.replace('.npy', '').split('_')
    
    try:
        date_str = parts[1]  # YYYYMMDD
        year = int(date_str[:4])
        
        return {
            'filename': filename,
            'date': date_str,
            'year': year,
        }
    except (IndexError, ValueError) as e:
        logger.warning(f"Could not parse filename {filename}: {e}")
        return {
            'filename': filename,
            'date': None,
            'year': None
        }


def format_index_json(output_dir: Path):
    """Reformat the index.json file for better readability."""
    index_file = output_dir / "index.json"
    
    if not index_file.exists():
        logger.warning(f"Warning: index.json not found at {index_file}")
        return
    
    with open(index_file, 'r') as f:
        data = json.load(f)
    
    # Write it back with proper formatting
    with open(index_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Formatted index.json for better readability")


def convert_to_mds(
    input_dir: str,
    output_dir: str,
    year_filter: Optional[int] = None,
    shard_size_mb: int = 64,
    compression: str = None, 
    ecg_dtype: str = 'ndarray:float32'
):
    """
    Convert ECG .npy files to MDS format.
    
    Args:
        input_dir: Directory containing .npy files
        output_dir: Directory to save MDS files
        year_filter: Optional year to filter files (e.g., 2020)
        shard_size_mb: Target size of each shard in MB
        compression: Compression algorithm ('zstd', 'snappy', 'brotli', or None)
        ecg_dtype: ECG data type specification (e.g., 'ndarray:float32:12,5000')
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Channel labels for ECG data
    channel_labels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                     'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Find all .npy files
    npy_files = list(input_path.glob("*.npy"))
    logger.info(f"Found {len(npy_files)} .npy files")
    
    # Filter by year if specified
    if year_filter is not None:
        filtered_files = []
        for npy_file in npy_files:
            metadata = parse_filename(npy_file.name)
            if metadata.get('year') == year_filter:
                filtered_files.append(npy_file)
        npy_files = filtered_files
        logger.info(f"Filtered to {len(npy_files)} files from year {year_filter}")
    
    if len(npy_files) == 0:
        logger.warning("No files to process!")
        return
    
    # Define schema for MDS
    columns = {
        'filename': 'str',
        'date': 'str',
        'year': 'int32',
        'channel_labels': 'json',
        'ecg_data': ecg_dtype
    }

    # Create MDS writer with rich progress bar
    with MDSWriter(
        out=output_path.as_posix(),
        columns=columns,
        compression=compression,
        size_limit=str(shard_size_mb) + "mb"
    ) as writer:
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
        ) as progress:
            
            task_id = progress.add_task("[cyan]Converting to MDS", total=len(npy_files))
            
            # Process each file
            for npy_file in npy_files:
                try:
                    ecg_data = np.load(npy_file)
                except Exception as e:
                    logger.warning(f"Error loading {npy_file}: {e}")
                    ecg_data = None

                if ecg_data is not None:
                    # Parse filename metadata
                    metadata = parse_filename(npy_file.name)
                    
                    # Create sample dictionary
                    sample = {
                        'filename': metadata['filename'],
                        'date': metadata.get('date', ''),
                        'year': metadata.get('year', 0),
                        'channel_labels': channel_labels,
                        'ecg_data': ecg_data
                    }
                    
                    # Write sample to MDS
                    writer.write(sample)
                
                progress.update(task_id, advance=1)
    
    # Format the index.json file for better readability
    format_index_json(output_path)

    logger.info(f"\nConversion complete!")
    logger.info(f"Output saved to: {output_path}")
    
    # Save metadata
    metadata_file = output_path / "conversion_metadata.json"
    metadata_info = {
        'total_samples': len(npy_files),
        'year_filter': year_filter,
        'channel_labels': channel_labels,
        'ecg_dtype': ecg_dtype,
        'compression': compression,
        'shard_size_mb': shard_size_mb
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata_info, f, indent=2)
    
    logger.info(f"Metadata saved to: {metadata_file}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert ECG .npy files to MDS format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Required arguments
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing .npy files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save MDS files')

    # Optional arguments
    parser.add_argument('--year', type=int, default=None, help='Filter files by year (e.g., 2020)')
    parser.add_argument('--shard_size_mb', type=int, default=64, help='Target shard size in megabytes')
    parser.add_argument('--compression', type=str, default=None, help='Compression algorithm')
    parser.add_argument(
        '--ecg_dtype', 
        type=str, 
        default='ndarray:float32:12,5000',
        help='ECG data type specification (e.g., "ndarray:float32:12,5000", "ndarray:float32", "ndarray")'
    )

    args = parser.parse_args()
    RichPrinter.print_config(args, "Convert Configuration")
    
    return args


def main():
    args = parse_args()
    
    # Handle 'none' compression
    compression = None if args.compression == 'none' else args.compression
    
    convert_to_mds(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        year_filter=args.year,
        shard_size_mb=args.shard_size_mb,
        compression=compression,
        ecg_dtype=args.ecg_dtype
    )


if __name__ == "__main__":
    main()

"""
Found 45239 .npy files
  Converting to MDS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% (45239/45239) 0:01:47
Formatted index.json for better readability
"""