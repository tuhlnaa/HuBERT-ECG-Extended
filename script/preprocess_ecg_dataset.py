"""
ECG Dataset Processing Module - Recursive File Processing

This module provides functionality for processing ECG datasets by recursively
searching for .hea files in a directory structure and flattening the output
to a single output folder.
"""

import logging
import sys
import wfdb
import numpy as np
from pathlib import Path
from typing import Union, List, Generator
from tqdm import tqdm
from rich.logging import RichHandler

# Import custom modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils import ecg_preprocessing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", 
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


class ECGRecursiveProcessor:
    """
    A class for recursively processing ECG datasets with flattened output structure.
    
    This processor searches for WFDB format ECG files (.hea) recursively in a directory
    structure and processes them to a flattened output directory without preserving
    the original folder hierarchy.
    """
    
    def __init__(
        self, 
        root_path: Union[str, Path], 
        output_path: Union[str, Path],
        skip_existing: bool = True
    ) -> None:
        """
        Initialize the recursive ECG dataset processor.
        
        Args:
            root_path: Root directory to search for .hea files recursively
            output_path: Path for processed output files (flattened structure)
            skip_existing: Whether to skip already processed files
        """
        self.root_path = Path(root_path)
        self.output_path = Path(output_path)
        self.skip_existing = skip_existing
        
        # Validate paths
        if not self.root_path.exists():
            raise FileNotFoundError(f"Root path does not exist: {self.root_path}")
        
        if not self.root_path.is_dir():
            raise NotADirectoryError(f"Root path is not a directory: {self.root_path}")
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized recursive ECG processor: {self.root_path} -> {self.output_path}")
    
    
    def find_hea_files(self) -> Generator[Path, None, None]:
        """Recursively find all .hea files in the root directory."""
        logger.info(f"Searching for .hea files recursively in: {self.root_path}")
        
        # Use rglob to recursively find all .hea files
        hea_files = list(self.root_path.rglob("*.hea"))
        
        logger.info(f"Found {len(hea_files)} .hea files")
        
        for hea_file in hea_files:
            yield hea_file
    
    
    def _generate_output_filename(self, hea_file_path: Path) -> str:
        """
        Generate standardized output filename based on the input .hea file.
        
        Args:
            hea_file_path: Path to the input .hea file
            
        Returns:
            Output filename following the pattern: original_name.hea.npy
        """
        # Get the filename without extension and add .hea.npy
        base_name = hea_file_path.stem  # Gets filename without .hea extension
        return f"{base_name}.hea.npy"
    
    
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
    
    
    def _process_single_file(self, hea_file_path: Path, output_file_path) -> bool:
        """Process a single ECG file."""
        try:
            # # Generate output filename and handle conflicts
            # output_filename = self._generate_output_filename(hea_file_path)
            # output_filename = self._handle_filename_conflicts(output_filename, hea_file_path)
            # output_filepath = self.output_path / output_filename
            
            # # Skip if file already exists and skip_existing is True
            # if self.skip_existing and output_filepath.exists():
            #     logger.debug(f"Skipping existing file: {output_filename}")
            #     return True
            
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
            
            # Save processed signal
            #np.save(output_filepath, processed_signal)
            np.save(output_file_path, processed_signal)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {hea_file_path.relative_to(self.root_path)}: {e}")
            return False
    
    
    def process_all_files(self) -> dict:
        """
        Process all .hea files found recursively in the root directory.
        
        Returns:
            Dictionary with processing statistics
        """
        # Find all .hea files
        hea_files = list(self.find_hea_files())
        
        if not hea_files:
            logger.warning("No .hea files found in the directory structure")
            return {"processed": 0, "failed": 0, "skipped": 0, "total_found": 0}
        
        logger.info(f"Starting processing of {len(hea_files)} ECG files")
        
        processed_count = 0
        failed_count = 0
        skipped_count = 0
        
        for hea_file_path in tqdm(hea_files, desc="Processing ECG files"):
            # Check if output already exists for skipping count
            output_filename = self._generate_output_filename(hea_file_path)
            output_filepath = self._handle_filename_conflicts(output_filename, hea_file_path)
            output_filepath = self.output_path / output_filename
            
            # if self.skip_existing and output_filepath.exists():
            if output_filepath.exists():
                skipped_count += 1
                continue
        
            success = self._process_single_file(hea_file_path, output_filepath)
            if success:
                processed_count += 1
            else:
                failed_count += 1
        
        stats = {
            "processed": processed_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "total_found": len(hea_files)
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


def main() -> None:
    """Main execution function for recursive ECG dataset processing."""
    # Configuration
    ROOT_PATH = Path(r"D:\Kai\ECG-dataset\G12EC\ptb")  # Change this to your root directory
    OUTPUT_PATH = Path("./output")  # Change this to your desired output directory
    
    try:
        # Initialize recursive processor
        processor = ECGRecursiveProcessor(
            root_path=ROOT_PATH,
            output_path=OUTPUT_PATH,
            skip_existing=True
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
        for directory, count in list(summary['files_per_directory'].items())[:3]:  # Show first 10
            logger.info(f"  {directory}: {count} files")
        
        if len(summary['files_per_directory']) > 3:
            logger.info(f"  ... and {len(summary['files_per_directory']) - 3} more directories")
        
        # Process all files
        logger.info("\n" + "=" * 50)
        logger.info("STARTING PROCESSING")
        logger.info("=" * 50)
        
        stats = processor.process_all_files()

        if stats['failed'] > 0:
            logger.warning(f"{stats['failed']} files failed processing")
        
        logger.info(f"\nAll processed files saved to: {OUTPUT_PATH}")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()