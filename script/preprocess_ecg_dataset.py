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
from typing import Dict, Union, Generator, Tuple, List
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
    Recursively processes ECG datasets in WFDB format with multiprocessing support.
    
    Searches for WFDB format ECG files (.hea) recursively and processes them to
    a flattened output directory structure.
    """
    
    def __init__(
        self,
        root_path: Union[str, Path],
        output_path: Union[str, Path],
        skip_existing: bool = True,
        n_processes: int = None,
    ) -> None:
        """
        Initialize the ECG dataset processor.
        
        Args:
            root_path: Root directory to search for .hea files recursively
            output_path: Output directory for processed files (flattened structure)
            skip_existing: Whether to skip already processed files
            n_processes: Number of processes to use (defaults to CPU count)
        
        Raises:
            FileNotFoundError: If root_path does not exist
            NotADirectoryError: If root_path is not a directory
        """
        self.root_path = Path(root_path)
        self.output_path = Path(output_path)
        self.skip_existing = skip_existing
        self.n_processes = self._validate_process_count(n_processes)
        
        logger.info(f"Using {self.n_processes} processes for ECG file processing")
        
        self._validate_paths()
        self.output_path.mkdir(parents=True, exist_ok=True)
    

    def _validate_process_count(self, n_processes: int) -> int:
        """Validate and normalize process count."""
        if n_processes is None:
            return mp.cpu_count()
        return max(1, min(n_processes, mp.cpu_count()))
    

    def _validate_paths(self) -> None:
        """Validate input paths."""
        if not self.root_path.exists():
            raise FileNotFoundError(f"Root path does not exist: {self.root_path}")
        
        if not self.root_path.is_dir():
            raise NotADirectoryError(f"Root path is not a directory: {self.root_path}")
    

    def _find_hea_files(self) -> List[Path]:
        """Recursively find all .hea files in the root directory."""
        return list(self.root_path.rglob("*.hea"))
    
    
    def _resolve_output_filename(self, hea_file_path: Path) -> str:
        """
        Resolve output filename, handling conflicts when necessary.
        
        Args:
            hea_file_path: Path to the input .hea file
            
        Returns:
            Unique output filename
        """
        base_filename = f"{hea_file_path.stem}.hea.npy"
        output_filepath = self.output_path / base_filename
        
        # No conflict or skip_existing is True
        if not output_filepath.exists() or self.skip_existing:
            return base_filename
        
        # Resolve conflict using parent directory name
        base_name = hea_file_path.stem
        parent_name = hea_file_path.parent.name
        
        # Try with parent directory name
        new_filename = f"{base_name}_{parent_name}.hea.npy"
        if not (self.output_path / new_filename).exists():
            logger.warning(f"Filename conflict resolved: {base_filename} -> {new_filename}")
            return new_filename
        
        # Use counter if still conflicts
        counter = 1
        while (self.output_path / f"{base_name}_{parent_name}_{counter}.hea.npy").exists():
            counter += 1
        
        final_filename = f"{base_name}_{parent_name}_{counter}.hea.npy"
        logger.warning(f"Filename conflict resolved: {base_filename} -> {final_filename}")
        return final_filename
    

    def _prepare_processing_tasks(self, hea_files: List[Path]) -> List[Tuple[Path, Path]]:
        """
        Prepare processing tasks from discovered files.
        
        Args:
            hea_files: List of .hea file paths
            
        Returns:
            List of (input_path, output_path) tuples for files to process
        """
        tasks = []
        
        for hea_file_path in hea_files:
            output_filename = self._resolve_output_filename(hea_file_path)
            output_filepath = self.output_path / output_filename
            
            # Skip if file exists and skip_existing is True
            if self.skip_existing and output_filepath.exists():
                continue
            
            tasks.append((hea_file_path, output_filepath))
        
        return tasks
    
    def process_all_files(self) -> Dict[str, int]:
        """
        Process all .hea files found recursively using multiprocessing.
        
        Returns:
            Dictionary with processing statistics:
                - processed: Number of successfully processed files
                - failed: Number of failed files
                - skipped: Number of skipped files
                - total_found: Total files discovered
        """
        logger.info("Discovering .hea files...")
        hea_files = self._find_hea_files()
        total_files_found = len(hea_files)
        
        if total_files_found == 0:
            logger.warning("No .hea files found in the directory structure")
            return {"processed": 0, "failed": 0, "skipped": 0, "total_found": 0}
        
        logger.info(f"Found {total_files_found} .hea files")
        logger.info("Preparing processing tasks...")
        tasks = self._prepare_processing_tasks(hea_files)
        
        skipped_count = total_files_found - len(tasks)
        
        if len(tasks) == 0:
            logger.info(f"All {total_files_found} files already exist and were skipped")
            return {
                "processed": 0,
                "failed": 0,
                "skipped": skipped_count,
                "total_found": total_files_found,
            }
        
        logger.info(
            f"Processing {len(tasks)} ECG files using {self.n_processes} processes"
        )
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} files that already exist")
        
        return self._process_with_multiprocessing(tasks, skipped_count, total_files_found)
    

    def _process_with_multiprocessing(
        self, tasks: List[Tuple[Path, Path]], skipped_count: int, total_files_found: int
    ) -> Dict[str, int]:
        """Execute processing tasks using multiprocessing with progress tracking."""
        processed_count = 0
        failed_count = 0
        failed_files = []
        
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeElapsedColumn(),
            ) as progress:
                
                task_id = progress.add_task("[green]Processing ECG files", total=len(tasks))
                
                futures = [
                    executor.submit(process_single_ecg_file, task) for task in tasks
                ]
                
                for future in as_completed(futures):
                    success, filename, error_msg = future.result()
                    
                    if success:
                        processed_count += 1
                    else:
                        failed_count += 1
                        failed_files.append((filename, error_msg))
                        logger.error(error_msg)
                    
                    progress.update(task_id, advance=1)
        
        self._log_failed_files(failed_files, failed_count)
        
        stats = {
            "processed": processed_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "total_found": total_files_found,
        }
        
        logger.info(f"Processing complete: {stats}")
        return stats
    

    def _log_failed_files(self, failed_files: List[Tuple[str, str]], failed_count: int) -> None:
        """Log summary of failed files."""
        if not failed_files:
            return
        
        logger.error(f"{failed_count} files failed processing:")
        for filename, error_msg in failed_files[:3]:
            logger.error(f"  {filename}: {error_msg}")
        
        if len(failed_files) > 3:
            logger.error(f"  ... and {len(failed_files) - 3} more failures")
    

    def get_file_summary(self) -> Dict[str, Union[int, Dict[str, int]]]:
        """
        Get a summary of discovered files without processing.
        
        Returns:
            Dictionary containing:
                - total_files: Total number of .hea files found
                - directories: Number of directories containing files
                - files_per_directory: Mapping of directory paths to file counts
        """
        hea_files = self._find_hea_files()
        
        dir_counts = {}
        for hea_file in hea_files:
            parent_dir = str(hea_file.parent.relative_to(self.root_path))
            dir_counts[parent_dir] = dir_counts.get(parent_dir, 0) + 1
        
        return {
            "total_files": len(hea_files),
            "directories": len(dir_counts),
            "files_per_directory": dir_counts,
        }


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for ECG dataset processing."""
    parser = argparse.ArgumentParser(
        description="Recursive ECG dataset processing with multiprocessing support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--root-path", type=Path, required=True, help="Root directory containing ECG dataset files")
    parser.add_argument("--output-path", type=Path, default=Path("./dataset"), help="Output directory for processed files")
    parser.add_argument("--n-processes", type=int, default=None, help="Number of processes for multiprocessing (defaults to CPU count)")
    
    return parser.parse_args()


def main() -> None:
    """Main execution function for ECG dataset processing."""
    args = parse_arguments()
    
    try:
        processor = ECGRecursiveProcessor(
            root_path=args.root_path,
            output_path=args.output_path,
            skip_existing=True,
            n_processes=args.n_processes,
        )
        
        logger.info("Analyzing directory structure...")
        summary = processor.get_file_summary()
        
        if summary["total_files"] == 0:
            logger.warning("No .hea files found. Exiting.")
            return
        
        stats = processor.process_all_files()
        
        if stats["failed"] > 0:
            logger.warning(f"{stats['failed']} files failed processing")
        
        logger.info(f"\nAll processed files saved to: {args.output_path}")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
