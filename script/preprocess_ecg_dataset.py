"""
ECG Dataset Processing Module - Recursive File Processing with Multiprocessing

This module provides functionality for processing ECG datasets by recursively
searching for .hea files in a directory structure and flattening the output
to a single output folder. Now supports multiprocessing for faster processing.

Uusage:
python ./script/preprocess_ecg_dataset.py --root-path "/path/to/ecg/data" --output-path "/path/to/output" --n-processes 10
"""

import argparse
import logging
import sys
import wfdb

import numpy as np
import multiprocessing as mp

from pathlib import Path
from typing import Union, Generator, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn

# Import custom modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from HuBert_ECG.utils import ecg_preprocessing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", 
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


def process_single_ecg_file(args: Tuple[Path, Path]) -> Tuple[bool, str, str]:
    """
    Worker function for processing a single ECG file.
    This function is designed to be used with multiprocessing.
    
    Args:
        args: Tuple containing (hea_file_path, output_file_path)
        
    Returns:
        Tuple of (success, relative_path, error_message)
    """
    hea_file_path, output_file_path = args
    
    try:
        # Read WFDB signal (use the path without .hea extension)
        signal_path = str(hea_file_path.with_suffix(''))
        signal, metadata = wfdb.rdsamp(signal_path)

        # Transpose signal for channel-first format (common in PyTorch)
        signal = signal.T
        
        # Handle NaN values
        if np.isnan(signal).any():
            logger.warning(f"NaN values found in {hea_file_path.name}, replacing with zeros")
            signal = np.nan_to_num(signal, nan=0.0)
        
        # Apply preprocessing
        sampling_rate = metadata['fs']
        processed_signal = ecg_preprocessing(signal, sampling_rate)
        processed_signal = processed_signal.astype(np.float32)

        # Save processed signal, e.g. shape=(38400, 12), float64
        np.save(output_file_path, processed_signal)
        
        return True, str(hea_file_path.name), ""
        
    except Exception as e:
        error_msg = f"Failed to process {hea_file_path.name}: {str(e)}"
        return False, str(hea_file_path.name), error_msg


class ECGRecursiveProcessor:
    """
    A class for recursively processing ECG datasets with flattened output structure.
    Now supports multiprocessing for improved performance.
    
    This processor searches for WFDB format ECG files (.hea) recursively in a directory
    structure and processes them to a flattened output directory without preserving
    the original folder hierarchy.
    """
    
    def __init__(
        self, 
        root_path: Union[str, Path], 
        output_path: Union[str, Path],
        skip_existing: bool = True,
        n_processes: int = None,
    ) -> None:
        """
        Initialize the recursive ECG dataset processor.
        
        Args:
            root_path: Root directory to search for .hea files recursively
            output_path: Path for processed output files (flattened structure)
            skip_existing: Whether to skip already processed files
            n_processes: Number of processes to use
            detailed_logging: Whether to show detailed logging during processing
        """
        self.root_path = Path(root_path)
        self.output_path = Path(output_path)
        self.skip_existing = skip_existing
        self.n_processes = max(1, min(n_processes, mp.cpu_count()))
        
        logger.info(f"Using {self.n_processes} processes for ECG file processing")
        
        # Validate paths
        if not self.root_path.exists():
            raise FileNotFoundError(f"Root path does not exist: {self.root_path}")
        
        if not self.root_path.is_dir():
            raise NotADirectoryError(f"Root path is not a directory: {self.root_path}")

        self.output_path.mkdir(parents=True, exist_ok=True)

    
    def find_hea_files(self) -> Generator[Path, None, None]:
        """Recursively find all .hea files in the root directory."""
        # Use rglob to recursively find all .hea files
        hea_files = list(self.root_path.rglob("*.hea"))
        
        for hea_file in hea_files:
            yield hea_file
    
    
    def _handle_filename_conflicts(self, output_filename: str, hea_file_path: Path) -> str:
        """Handle potential filename conflicts when flattening directory structure."""
        output_filepath = self.output_path / output_filename
        
        if not output_filepath.exists() or self.skip_existing:
            return output_filename
        
        # If file exists, add parent directory info to make it unique
        counter = 1
        base_name = Path(output_filename).stem.replace('.hea', '')
        parent_name = hea_file_path.parent.name
        
        # Try with parent directory name first
        new_filename = f"{base_name}_{parent_name}.hea.npy"
        new_filepath = self.output_path / new_filename
        
        if not new_filepath.exists():
            logger.warning(f"Filename conflict resolved: {output_filename} -> {new_filename}")
            return new_filename
        
        # If still conflicts, use counter
        while (self.output_path / f"{base_name}_{parent_name}_{counter}.hea.npy").exists():
            counter += 1
        
        final_filename = f"{base_name}_{parent_name}_{counter}.hea.npy"
        logger.warning(f"Filename conflict resolved: {output_filename} -> {final_filename}")
        return final_filename
    
    
    def _prepare_processing_tasks(self) -> List[Tuple[Path, Path]]:
        """
        Prepare all processing tasks by finding files and handling conflicts.
        
        Returns:
            List of tuples (hea_file_path, output_file_path)
        """
        hea_files = list(self.find_hea_files())
        tasks = []
        
        for hea_file_path in hea_files:
            # Handle filename conflicts
            output_filename = f"{hea_file_path.stem}.hea.npy"
            output_filename = self._handle_filename_conflicts(output_filename, hea_file_path)
            output_filepath = self.output_path / output_filename
            
            # Skip if file already exists and skip_existing is True
            if self.skip_existing and output_filepath.exists():
                continue
                
            tasks.append((hea_file_path, output_filepath))
        
        return tasks
    
    
    def process_all_files(self) -> dict:
        """
        Process all .hea files found recursively in the root directory using multiprocessing.
        
        Returns:
            Dictionary with processing statistics
        """
        # Find all .hea files and prepare tasks
        logger.info("Preparing processing tasks...")
        tasks = self._prepare_processing_tasks()
        total_files_found = len(list(self.find_hea_files()))
        
        if total_files_found == 0:
            logger.warning("No .hea files found in the directory structure")
            return {"processed": 0, "failed": 0, "skipped": 0, "total_found": 0}
        
        skipped_count = total_files_found - len(tasks)
        
        if len(tasks) == 0:
            logger.info(f"All {total_files_found} files already exist and were skipped")
            return {
                "processed": 0, 
                "failed": 0, 
                "skipped": skipped_count, 
                "total_found": total_files_found
            }
        
        logger.info(f"Starting multiprocessing of {len(tasks)} ECG files using {self.n_processes} processes")
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} files that already exist")
        
        processed_count = 0
        failed_count = 0
        failed_files = []
        
        # Use ProcessPoolExecutor for multiprocessing with rich progress bar
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeElapsedColumn(),
            ) as progress:
                
                # Add progress task
                task_id = progress.add_task(f"[green]Processing ECG files", total=len(tasks))
                
                # Submit all tasks
                futures = [executor.submit(process_single_ecg_file, task) for task in tasks]
                
                # Process results as they complete
                for future in as_completed(futures):
                    success, filename, error_msg = future.result()
                    
                    if success:
                        processed_count += 1
                    else:
                        failed_count += 1
                        failed_files.append((filename, error_msg))
                        logger.error(error_msg)
                    
                    progress.update(task_id, advance=1)
        
        # Log failed files summary
        if failed_files:
            logger.error(f"{failed_count} files failed processing:")
            for filename, error_msg in failed_files[:3]:  # Show first 3 errors
                logger.error(f"  {filename}: {error_msg}")
            if len(failed_files) > 3:
                logger.error(f"  ... and {len(failed_files) - 3} more failures")
        
        stats = {
            "processed": processed_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "total_found": total_files_found
        }
        
        logger.info(f"Processing complete: {stats}")
        return stats
    
    
    def get_file_summary(self) -> dict:
        """Get a summary of files found without processing them."""
        hea_files = list(self.find_hea_files())
        
        # Group files by directory for summary
        dir_counts = {}
        for hea_file in hea_files:
            parent_dir = hea_file.parent.relative_to(self.root_path)
            dir_counts[str(parent_dir)] = dir_counts.get(str(parent_dir), 0) + 1
        
        return {
            "total_files": len(hea_files),
            "directories": len(dir_counts),
            "files_per_directory": dir_counts
        }


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for ECG dataset processing."""
    parser = argparse.ArgumentParser(
        description="Recursive ECG dataset processing with multiprocessing support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--root-path", type=Path, help="Root directory path containing ECG dataset files")
    parser.add_argument("--output-path", type=Path, default=Path("./dataset"), help="Output directory path for processed files")
    parser.add_argument("--n-processes", type=int, default=8, help="Number of processes for multiprocessing")
    
    return parser.parse_args()


def main() -> None:
    """Main execution function for recursive ECG dataset processing with multiprocessing."""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Initialize recursive processor with multiprocessing support
        processor = ECGRecursiveProcessor(
            root_path=args.root_path,
            output_path=args.output_path,
            skip_existing=True,
            n_processes=args.n_processes,
        )
       
        # Get file summary
        logger.info("Analyzing directory structure...")
        summary = processor.get_file_summary()
       
        logger.info("=" * 50)
        logger.info("FILE DISCOVERY SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total .hea files found: {summary['total_files']}")
        logger.info(f"Directories containing files: {summary['directories']}")
       
        if summary['total_files'] == 0:
            logger.warning("No .hea files found. Exiting.")
            return
       
        # Show files per directory (limit output for readability)
        logger.info("\nFiles per directory:")
        for directory, count in list(summary['files_per_directory'].items())[:3]:  # Show first 3
            logger.info(f"  {directory}: {count} files")
       
        if len(summary['files_per_directory']) > 3:
            logger.info(f"  ... and {len(summary['files_per_directory']) - 3} more directories")
       
        stats = processor.process_all_files()

        if stats['failed'] > 0:
            logger.warning(f"{stats['failed']} files failed processing")
       
        logger.info(f"\nAll processed files saved to: {args.output_path}")
       
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()