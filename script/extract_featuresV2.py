"""
ECG Feature Extraction Module - Morphological and Latent Feature Dumping

This module provides functionality for extracting features from ECG datasets
for HuBERT-ECG training. Supports both morphological feature extraction 
(iteration 1) and latent feature extraction from trained HuBERT models 
(iterations 2+). Features are saved as numpy arrays for downstream tasks.

Usage:
# Extract morphological features (iteration 1)
python script/extract_features.py 1 "/path/to/dataframe.csv" "/path/to/ecg/data" "/path/to/output" 0.0 1.0 --mfcc_only --sample_rate 500

# Extract latent features (iteration 2+)
python script/extract_features.py 2 "/path/to/dataframe.csv" "/path/to/ecg/data" "/path/to/output" 0.0 1.0 --hubert_path "/path/to/model.pt" --output_layer 2 --batch_size 32
"""

import logging
import sys
import time
import torch

import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from rich.logging import RichHandler
from torch.utils.data import DataLoader
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Import custom modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from HuBert_ECG.config import create_dumping_parser, init_seeds
from HuBert_ECG.dataset import ECGDataset
from HuBert_ECG.ecg_features import Config, ECGDataProcessor, FeatureExtractorFactory
from HuBert_ECG.hubert_ecg import HubertECG, HuBERTECGConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", 
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for feature extraction."""
    dataset_csv_path: Path
    ecg_dir: Path
    output_dir: Path
    batch_size: int = 8
    data_slice: Tuple[float, float] = (0.0, 1.0)
    downsampling_factor: int = 5
    num_workers: int = 0
    save_metadata_csv: bool = False
    iteration_id: Optional[int] = None


# Add this module-level function (needs to be at module level for pickling)
def process_single_record_features(
    task: Tuple
) -> Tuple[bool, str, Optional[str]]:
    """
    Process a single record for feature extraction.
    
    Args:
        task: Tuple of (record_dict, input_dir, output_dir, feature_mode, 
              sample_rate, base_sample_rate, config_dict, skip_existing, max_samples, device_str)
    
    Returns:
        Tuple of (success, filename, error_message)
    """
    (record_dict, input_dir, output_dir, feature_mode, sample_rate, 
     base_sample_rate, config_dict, skip_existing, max_samples, device_str) = task
    
    filename = record_dict['filename']
    
    try:
        # Create local instances (each process needs its own)
        # Use the assigned device for this task
        device = torch.device(device_str)
        
        # Reconstruct config from dict
        from types import SimpleNamespace
        config = SimpleNamespace(**config_dict)
        
        # Create processor and feature extractor for this worker
        processor = ECGDataProcessor(config, sample_rate, base_sample_rate)
        feature_extractor = FeatureExtractorFactory.create(
            feature_mode, sample_rate, device
        )
        
        input_path = Path(input_dir) / filename
        output_path = Path(output_dir) / filename
        
        # Skip if file exists
        if skip_existing and output_path.exists():
            return (True, filename, None)  # Success but skipped
        
        # Load and preprocess data
        data = processor.load_and_preprocess(input_path, max_samples, filename)
        
        if data is None:
            return (False, filename, f"Failed to load and preprocess data")
        
        # Process data into shards
        shards = processor.process_to_shards(data)
        shards = torch.from_numpy(shards).to(device, dtype=torch.float32)
        
        # Extract features
        features = feature_extractor.extract(shards)
        
        # Validate output
        expected_n_shards = (data.shape[0] * data.shape[1]) // config.compression_factor
        expected_dim = FeatureExtractorFactory.get_feature_dim(feature_mode)
        
        if len(features) != expected_n_shards:
            return (False, filename, 
                   f"Expected {expected_n_shards} feature vectors, got {len(features)}")
        
        if len(features[0]) != expected_dim:
            return (False, filename,
                   f"Expected {expected_dim} features, got {len(features[0])}")
        
        # Save features
        np.save(output_path, features.astype(np.float32))
        
        return (True, filename, None)
        
    except Exception as e:
        return (False, filename, str(e))

class ECGFeatureExtractor:
    """Main class for extracting features from ECG records."""
    
    def __init__(
        self, 
        feature_mode: str,
        sample_rate: int, 
        base_sample_rate: int = 500, 
        device = ["cuda:0", "cuda:1"],  # Now accepts list of device strings
        skip_existing: bool = True,
        n_processes: int = 5,
    ):
        self.devices = device
        self.skip_existing = skip_existing
        self.n_processes = n_processes
        
        logger.info(f"Using {self.n_processes} processes for feature extraction")
        logger.info(f"Distributing work across devices: {self.devices}")

        # Validate sampling rate
        if sample_rate not in Config.SAMPLING:
            error_msg = (f"Unsupported sample_rate: {sample_rate}. "
                        f"Must be one of {list(Config.SAMPLING.keys())}")
            raise ValueError(error_msg)
        
        self.config = Config.SAMPLING[sample_rate]
        self.sample_rate = sample_rate
        self.base_sample_rate = base_sample_rate


    def validate_features(self, features: List, expected_n_shards: int, 
                         feature_mode: str) -> None:
        """Validate extracted features dimensions."""
        expected_dim = FeatureExtractorFactory.get_feature_dim(feature_mode)
        
        assert len(features) == expected_n_shards, (
            f"Expected {expected_n_shards} feature vectors, got {len(features)}"
        )
        assert len(features[0]) == expected_dim, (
            f"Expected {expected_dim} features for mode '{feature_mode}', "
            f"got {len(features[0])}"
        )


    def _prepare_processing_tasks(
        self, dataframe: pd.DataFrame, input_dir: Path, output_dir: Path,
        feature_mode: str, max_samples: int
    ) -> List[Tuple]:
        """
        Prepare processing tasks from dataframe records.
        Distributes devices evenly across tasks in round-robin fashion.
        
        Returns:
            List of task tuples for multiprocessing
        """
        tasks = []
        
        # Convert config to dict for pickling
        config_dict = vars(self.config)
        
        task_index = 0
        for record in dataframe.itertuples(index=False):
            # Convert record to dict for pickling
            record_dict = record._asdict()
            
            output_path = output_dir / record_dict['filename']
            
            # Skip if file exists and skip_existing is True
            if self.skip_existing and output_path.exists():
                continue
            
            # Assign device in round-robin fashion
            device_str = self.devices[task_index % len(self.devices)]
            
            task = (
                record_dict,
                str(input_dir),
                str(output_dir),
                feature_mode,
                self.sample_rate,
                self.base_sample_rate,
                config_dict,
                self.skip_existing,
                max_samples,
                device_str  # Add device assignment
            )
            tasks.append(task)
            task_index += 1
        
        return tasks


    def _process_with_multiprocessing(
        self, tasks: List[Tuple], total_records: int
    ) -> Dict[str, int]:
        """Execute processing tasks using multiprocessing with progress tracking."""
        processed_count = 0
        failed_count = 0
        skipped_count = total_records - len(tasks)
        failed_records = []
        
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeElapsedColumn(),
            ) as progress:
                
                task_id = progress.add_task(
                    "[green]Extracting features", 
                    total=len(tasks)
                )
                
                futures = [
                    executor.submit(process_single_record_features, task) 
                    for task in tasks
                ]
                
                for future in as_completed(futures):
                    success, filename, error_msg = future.result()
                    
                    if success:
                        processed_count += 1
                    else:
                        failed_count += 1
                        failed_records.append((filename, error_msg))
                        logger.error(f"Failed: {filename} - {error_msg}")
                    
                    progress.update(task_id, advance=1)
        
        self._log_failed_records(failed_records, failed_count)
        
        return {
            "processed": processed_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "total": total_records,
        }


    def extract_batch(
        self, dataframe: pd.DataFrame, input_dir: Path, 
        output_dir: Path, feature_mode: str, sample_rate: int,
        max_samples: int = 2500, use_multiprocessing: bool = True
    ) -> Dict[str, int]:
        """
        Extract morphological features for all records in dataframe.
        
        Args:
            dataframe: DataFrame containing records to process
            input_dir: Directory containing input files
            output_dir: Directory for output files
            feature_mode: Feature extraction mode
            sample_rate: Sampling rate
            max_samples: Maximum samples to process
            use_multiprocessing: Whether to use multiprocessing (default: True)
        
        Returns:
            Dictionary with extraction statistics:
                - processed: Number of successfully processed records
                - failed: Number of failed records
                - skipped: Number of skipped records
                - total: Total records in dataframe
        """
        total_records = len(dataframe)
        
        if total_records == 0:
            logger.warning("No records found in dataframe")
            return {"processed": 0, "failed": 0, "skipped": 0, "total": 0}
        
        logger.info(f"Starting feature extraction for {total_records} records")
        
        # Use multiprocessing if requested and n_processes > 1
        if use_multiprocessing and self.n_processes > 1:
            tasks = self._prepare_processing_tasks(
                dataframe, input_dir, output_dir, feature_mode, max_samples
            )
            
            if len(tasks) == 0:
                logger.info(f"All {total_records} files already exist and were skipped")
                return {
                    "processed": 0,
                    "failed": 0,
                    "skipped": total_records,
                    "total": total_records,
                }
            
            skipped_count = total_records - len(tasks)
            logger.info(
                f"Processing {len(tasks)} records using {self.n_processes} processes"
            )
            if skipped_count > 0:
                logger.info(f"Skipping {skipped_count} files that already exist")
            
            return self._process_with_multiprocessing(tasks, total_records)


    def _log_failed_records(
        self, failed_records: List[Tuple[str, str]], 
        failed_count: int
    ) -> None:
        """Log summary of failed records."""
        if not failed_records:
            return
        
        logger.error(f"{failed_count} records failed feature extraction:")
        for filename, error_msg in failed_records[:3]:
            logger.error(f"  {filename}: {error_msg}")
        
        if len(failed_records) > 3:
            logger.error(f"  ... and {len(failed_records) - 3} more failures")


class LatentFeatureExtractor:
    """Handles extraction of latent features from HuBERT model."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        layer_idx: int,
        config: ExtractionConfig,
    ):
        """
        Initialize the feature extractor.
        
        Args:
            model: HuBERT model for encoding ECGs
            layer_idx: Hidden layer index to extract features from (0-indexed)
            config: Configuration for extraction process
        """
        self.model = model
        self.layer_idx = layer_idx
        self.config = config
        self.device = next(model.parameters()).device
        
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()
    

    def extract_and_save(self) -> None:
        """Execute the complete feature extraction pipeline."""
        # Setup dataset and dataloader
        dataloader = self._create_dataloader()
        
        # Save metadata if requested
        if self.config.save_metadata_csv:
            self._save_metadata_csv(dataloader.dataset.ecg_dataframe)
        
        # Extract and save features
        self._process_batches(dataloader)
    

    def _create_dataloader(self) -> DataLoader:
        """Create dataloader with appropriate settings."""
        dataset = ECGDataset(
            path_to_dataset_csv=self.config.dataset_csv_path,
            ecg_dir_path=self.config.ecg_dir,
            downsampling_factor=self.config.downsampling_factor,
            pretrain=False,
            encode=True,
        )
        
        # Apply data slicing
        start_perc, end_perc = self.config.data_slice
        start_idx = int(start_perc * len(dataset))
        end_idx = int(end_perc * len(dataset)) + 1
        dataset.ecg_dataframe = dataset.ecg_dataframe.iloc[start_idx:end_idx].copy()

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=dataset.collate,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
    

    def _process_batches(self, dataloader: DataLoader) -> None:
        """Process all batches and extract features."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
        ) as progress:
            
            task_id = progress.add_task("[green]Extracting features", total=len(dataloader))
            
            for batch_idx, (ecgs, ecg_filenames) in enumerate(dataloader):
                ecgs = ecgs.to(self.device, non_blocking=True)
                
                # Extract features
                features = self._extract_features_from_batch(ecgs)
                
                # Validate dimensions
                self._validate_features(features, ecg_filenames)
                
                # Save features
                self._save_batch_features(features, ecg_filenames)
                
                progress.update(task_id, advance=1)
        

    def _extract_features_from_batch(self, ecgs: torch.Tensor) -> np.ndarray:
        """Extract features from a batch of ECGs."""
        with torch.no_grad():
            outputs = self.model(
                ecgs,
                attention_mask=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True
            )
        
        return outputs['hidden_states'][self.layer_idx].cpu().numpy()
    

    def _validate_features(
        self,
        features: np.ndarray,
        ecg_filenames: list,
        expected_seq_len: int = 93
    ) -> None:
        """Validate extracted feature dimensions."""
        batch_size, seq_len, hidden_size = features.shape
        expected_hidden_size = self.model.config.hidden_size
        
        assert seq_len == expected_seq_len, (
            f"Unexpected sequence length: {seq_len} (expected {expected_seq_len})"
        )
        assert hidden_size == expected_hidden_size, (
            f"Unexpected hidden size: {hidden_size} (expected {expected_hidden_size})"
        )
        assert batch_size == len(ecg_filenames), (
            f"Batch size mismatch: {batch_size} != {len(ecg_filenames)}"
        )
    

    def _save_batch_features(
        self,
        features: np.ndarray,
        ecg_filenames: list
    ) -> None:
        """Save individual features for each ECG in the batch."""
        for feature, filename in zip(features, ecg_filenames):
            feature_path = self.config.output_dir / f"{Path(filename).stem}.npy"
            np.save(feature_path, feature)
    

    def _save_metadata_csv(self, dataframe: pd.DataFrame) -> None:
        """Save metadata CSV referencing extracted features."""
        start_perc, end_perc = self.config.data_slice
        perc_size = int((end_perc - start_perc) * 100)
        iteration_suffix = f"_it{self.config.iteration_id}" if self.config.iteration_id is not None else ""
        
        csv_path = self.config.output_dir / f"latent_{perc_size}perc_layer{self.layer_idx + 1}{iteration_suffix}.csv"
        dataframe.to_csv(csv_path, index=False)
        logger.info(f"Saved metadata CSV: {csv_path}")


class FeatureExtractionPipeline:
    """Main pipeline for ECG feature extraction."""
    
    def __init__(self, args):
        """Initialize pipeline with command-line arguments."""
        self.args = args
        self.device = args.device
        self.num_process = args.num_process
        self.input_dir = Path(args.input_dir)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skip_existing=not args.overwrite
    
    def run(self) -> None:
        """
        Execute feature extraction pipeline.
        
        Iteration 1: Extract morphological features
        Iteration 2+: Extract latent features from HuBERT model
        """
        if self.args.iteration == 1:
            self._extract_morphological_features()
        else:
            self._extract_latent_features()
    

    def _extract_morphological_features(self) -> None:
        """Extract morphological features using ECGFeatureExtractor."""
        logger.info("Extracting morphological features...")
        
        # Load and slice dataframe
        dataframe = self._load_and_slice_dataframe()
        
        # Extract features
        extractor = ECGFeatureExtractor(
            feature_mode=self.args.feature_mode,
            sample_rate=self.args.sample_rate,
            device=self.device, 
            skip_existing=self.skip_existing,
            n_processes=self.num_process
        )
        extractor.extract_batch(
            dataframe=dataframe,
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            feature_mode=self.args.feature_mode,
            sample_rate=self.args.sample_rate,
        )
    

    def _extract_latent_features(self) -> None:
        """Extract latent features from HuBERT model."""
        # Configure extraction
        config = ExtractionConfig(
            dataset_csv_path=self.args.df_path,
            ecg_dir=self.input_dir,
            output_dir=self.output_dir,
            batch_size=self.args.batch_size,
            data_slice=(self.args.subset_start, self.args.subset_end),
            save_metadata_csv=self.args.save_csv,
            iteration_id=self.args.iteration,
        )

        logger.info(
            f"Extracting latent features from layer {self.args.output_layer + 1} "
            f"of HuBERT encoder..."
        )

        # Load model
        logger.info("Loading HuBERT model...")
        checkpoint = torch.load(self.args.hubert_path, map_location='cpu', weights_only=False)
        model_config = checkpoint['model_config']
        model_config.conv_pos_batch_norm = False
        
        model = HubertECG(model_config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(self.device)
        model.eval()
        
        # Extract features
        extractor = LatentFeatureExtractor(model, self.args.output_layer, config)
        extractor.extract_and_save()
    

    def _load_and_slice_dataframe(self) -> pd.DataFrame:
        """Load dataframe and apply slicing."""
        logger.info("Loading dataframe...")
        dataframe = pd.read_csv(self.args.df_path)
        start_idx = int(self.args.subset_start * len(dataframe))
        end_idx = int(self.args.subset_end * len(dataframe)) + 1
        return dataframe.iloc[start_idx:end_idx]


def main():
    """Main entry point for feature extraction."""
    args = create_dumping_parser()
    args.device = args.device[0].split()
    init_seeds(seed=42)
    
    pipeline = FeatureExtractionPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
