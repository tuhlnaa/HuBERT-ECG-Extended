"""
ECG ADC CSV to NPY Converter - Recursive File Processing with Multiprocessing

This module converts ECG ADC data from CSV format to NumPy array format (.npy).
Supports two conversion formulas:
1. Standard: physical_value_mV = (ADC_value - baseline) / gain
2. Alternative: physical_value_mV = (ADC_value / gain) + adc_zero

Usage:
# Using standard formula (baseline)
python script/ecg_csv_to_npy.py --input-dir "/path/to/csv/data" --output-dir "/path/to/output" --gain 1000 --baseline 0 --n-processes 10

# Using alternative formula (adc_zero)
python script/ecg_csv_to_npy.py --input-dir "/path/to/csv/data" --output-dir "/path/to/output" --gain 1000 --adc-zero -0.5 --n-processes 10
"""

import argparse
import logging
import multiprocessing as mp
import numpy as np
import pandas as pd

from pathlib import Path
from rich.logging import RichHandler
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
from typing import Dict, Union, Tuple, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

# Expected ECG leads in order
EXPECTED_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def convert_adc_to_physical(
    adc_data: np.ndarray,
    gain: float,
    baseline: Optional[float] = None,
    adc_zero: Optional[float] = None
) -> np.ndarray:
    """
    Convert ADC values to physical values (mV) using one of two formulas.
    
    Args:
        adc_data: ADC values to convert
        gain: Gain value for conversion
        baseline: Baseline value for standard formula
        adc_zero: ADC zero value for alternative formula
    
    Returns:
        Physical values in mV
    """
    if baseline is not None:
        return (adc_data - baseline) / gain
    elif adc_zero is not None:
        return (adc_data / gain) + adc_zero
    else:
        raise ValueError("Either baseline or adc_zero must be provided")


def process_single_csv_file(args: Tuple[Path, Path, float, Optional[float], Optional[float]]) -> Tuple[bool, str, str]:
    """
    Worker function for processing a single CSV file.
    This function is designed to be used with multiprocessing.
    
    Args:
        args: Tuple containing (csv_file_path, output_file_path, gain, baseline, adc_zero)
        
    Returns:
        Tuple of (success, relative_path, error_message)
    """
    csv_file_path, output_file_path, gain, baseline, adc_zero = args
    
    try:
        df = pd.read_csv(csv_file_path)
        
        # Verify all leads are present
        if list(df.columns) != EXPECTED_LEADS:
            raise ValueError(f"CSV columns must be exactly: {EXPECTED_LEADS}")
        
        # Convert to numpy array and transpose to get shape (12, n_samples)
        adc_data = df.values.T
        
        # Convert ADC values to physical values (mV)
        physical_data = convert_adc_to_physical(adc_data, gain, baseline, adc_zero)
        physical_data = physical_data.astype(np.float32)

        np.save(output_file_path, physical_data)
        
        return True, str(csv_file_path.name), ""
        
    except Exception as e:
        error_msg = f"Failed to process {csv_file_path.name}: {str(e)}"
        return False, str(csv_file_path.name), error_msg


class ECGCSVConverter:
    """
    Recursively converts ECG CSV files to NPY format with multiprocessing support.
    
    Searches for CSV files recursively and processes them to a flattened or
    mirrored output directory structure.
    """
    
    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        gain: float,
        baseline: Optional[float] = None,
        adc_zero: Optional[float] = None,
        skip_existing: bool = True,
        n_processes: Optional[int] = None,
        flatten_output: bool = False,
    ) -> None:
        """
        Initialize the ECG CSV converter.
        
        Args:
            input_dir: Root directory to search for CSV files recursively
            output_dir: Output directory for processed files
            gain: Gain value for conversion
            baseline: Baseline value for standard formula
            adc_zero: ADC zero value for alternative formula
            skip_existing: Whether to skip already processed files
            n_processes: Number of processes to use (defaults to CPU count)
            flatten_output: If True, flatten output structure; if False, mirror input structure
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.gain = gain
        self.baseline = baseline
        self.adc_zero = adc_zero
        self.skip_existing = skip_existing
        self.n_processes = self._validate_process_count(n_processes)
        self.flatten_output = flatten_output
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._validate_conversion_params()
        self._validate_paths()

        logger.info(f"Using {self.n_processes} processes for CSV file processing")


    def _validate_process_count(self, n_processes: Optional[int]) -> int:
        """Validate and normalize process count."""
        if n_processes is None:
            return mp.cpu_count()
        return max(1, min(n_processes, mp.cpu_count()))
    

    def _validate_paths(self) -> None:
        """Validate input paths."""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")
        
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {self.input_dir}")
    
    
    def _validate_conversion_params(self) -> None:
        """Validate conversion parameters."""
        if self.baseline is None and self.adc_zero is None:
            raise ValueError("Either baseline or adc_zero must be provided")
        
        if self.baseline is not None and self.adc_zero is not None:
            raise ValueError("Only one of baseline or adc_zero should be provided")
    

    def process_all_files(self) -> Dict[str, int]:
        """Process all CSV files found recursively using multiprocessing."""
        csv_files = list(self.input_dir.rglob("*.csv"))
        total_files_found = len(csv_files)

        if total_files_found == 0:
            logger.warning("No CSV files found in the directory structure")
            return {"processed": 0, "failed": 0, "skipped": 0, "total_found": 0}
        
        logger.info(f"Found {total_files_found} CSV files")
        tasks = self._prepare_processing_tasks(csv_files)
        
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
            f"Processing {len(tasks)} CSV files using {self.n_processes} processes"
        )
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} files that already exist")
        
        return self._process_with_multiprocessing(tasks, skipped_count, total_files_found)
    

    def _prepare_processing_tasks(self, csv_files: List[Path]) -> List[Tuple[Path, Path, float, Optional[float], Optional[float]]]:
        """
        Prepare processing tasks from discovered files.
        """
        tasks = []
        
        for csv_file_path in csv_files:
            if self.flatten_output:
                # Flatten: all outputs in single directory
                output_filepath = self.output_dir / f"{csv_file_path.stem}.npy"
            else:
                # Mirror: maintain directory structure
                relative_path = csv_file_path.relative_to(self.input_dir)
                output_filepath = self.output_dir / relative_path.with_suffix('.npy')

            # Skip if file exists and skip_existing is True
            if self.skip_existing and output_filepath.exists():
                continue
            
            # Create output directory if needed (for mirrored structure)
            if not self.flatten_output:
                output_filepath.parent.mkdir(parents=True, exist_ok=True)
            
            tasks.append((csv_file_path, output_filepath, self.gain, self.baseline, self.adc_zero))
        
        return tasks
    

    def _process_with_multiprocessing(
        self, tasks: List[Tuple], skipped_count: int, total_files_found: int
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
                
                task_id = progress.add_task("[green]Processing CSV files", total=len(tasks))
                
                futures = [
                    executor.submit(process_single_csv_file, task) for task in tasks
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


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for ECG CSV conversion."""
    parser = argparse.ArgumentParser(
        description="Convert ECG ADC CSV files to .npy format with multiprocessing support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
   
    parser.add_argument("--input-dir", type=Path, required=True, help="Root directory containing CSV files")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"), help="Output directory for .npy files")
    parser.add_argument("--gain", type=float, required=True, help="Gain value for conversion")
    
    # Create mutually exclusive group for baseline/adc-zero
    conversion_group = parser.add_mutually_exclusive_group(required=True)
    conversion_group.add_argument("--baseline", type=float, help="Baseline value (for standard formula)")
    conversion_group.add_argument("--adc-zero", type=float, help="ADC zero value (for alternative formula)")
    
    parser.add_argument("--n-processes", type=int, default=None, help="Number of processes for multiprocessing (defaults to CPU count)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files (default: skip existing files)")
    parser.add_argument("--flatten", action="store_true", help="Flatten output directory structure (default: mirror input structure)")

    return parser.parse_args()


def main() -> None:
    """Main execution function for ECG CSV conversion."""
    args = parse_arguments()
    
    converter = ECGCSVConverter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        gain=args.gain,
        baseline=args.baseline,
        adc_zero=args.adc_zero,
        skip_existing=not args.overwrite,
        n_processes=args.n_processes,
        flatten_output=args.flatten,
    )
    
    stats = converter.process_all_files()
    
    if stats["failed"] > 0:
        logger.warning(f"{stats['failed']} files failed processing")
    
    logger.info(f"\nAll processed files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()