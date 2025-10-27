"""
ECG Feature Extraction Module - Morphological and Latent Feature Dumping

This module provides functionality for extracting features from ECG datasets
for HuBERT-ECG training. Supports both morphological feature extraction 
(iteration 1) and latent feature extraction from trained HuBERT models 
(iterations 2+). Features are saved as numpy arrays for downstream tasks.

Usage:
# Extract morphological features (iteration 1)
python ./script/extract_features.py 1 "/path/to/dataframe.csv" "/path/to/ecg/data" "/path/to/output" 0.0 1.0 --mfcc_only --samp_rate 500

# Extract latent features (iteration 2+)
python ./script/extract_features.py 2 "/path/to/dataframe.csv" "/path/to/ecg/data" "/path/to/output" 0.0 1.0 --hubert_path "/path/to/model.pt" --output_layer 2 --batch_size 32
"""

import logging
import sys
import torch

import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from rich.logging import RichHandler
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Tuple

# Import custom modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from HuBert_ECG.config import create_dumping_parser, init_seeds
from HuBert_ECG.dataset import ECGDataset
from HuBert_ECG.hubert_ecg import HuBERTECG, HuBERTECGConfig
from HuBert_ECG.ecg_features import ECGFeatureExtractor

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
        dataset = self._create_dataset()
        
        # Save metadata if requested
        if self.config.save_metadata_csv:
            self._save_metadata_csv(dataset.ecg_dataframe)
        
        # Create dataloader
        dataloader = self._create_dataloader(dataset)
        
        # Extract and save features
        self._process_batches(dataloader)
    

    def _create_dataset(self) -> 'ECGDataset':
        """Create and slice ECG dataset based on configuration."""
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
        
        return dataset
    

    def _create_dataloader(self, dataset: 'ECGDataset') -> DataLoader:
        """Create dataloader with appropriate settings."""
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
        for batch_idx, (ecgs, ecg_filenames) in enumerate(tqdm(dataloader, desc="Extracting features")):
            ecgs = ecgs.to(self.device, non_blocking=True)
            
            # Extract features
            features = self._extract_features_from_batch(ecgs)
            
            # Validate dimensions
            self._validate_features(features, ecg_filenames)
            
            # Save features
            self._save_batch_features(features, ecg_filenames)
            
            logger.info(f"Saved batch {batch_idx}: {features.shape}")
    

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dir = Path(args.in_dir)
        self.output_dir = Path(args.dest_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> None:
        """
        Execute feature extraction pipeline.
        
        Iteration 1: Extract morphological features
        Iteration 2+: Extract latent features from HuBERT model
        """
        if self.args.train_iteration == 1:
            self._extract_morphological_features()
        else:
            self._extract_latent_features()
    

    def _extract_morphological_features(self) -> None:
        """Extract morphological features using ECGFeatureExtractor."""
        logger.info("Extracting morphological features...")
        
        # Determine feature mode
        feature_mode = self._get_feature_mode()
        
        # Load and slice dataframe
        dataframe = self._load_and_slice_dataframe()
        
        # Extract features
        extractor = ECGFeatureExtractor(self.device)
        extractor.extract_batch(
            dataframe=dataframe,
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            feature_mode=feature_mode,
            sample_rate=self.args.samp_rate,
        )
    

    def _extract_latent_features(self) -> None:
        """Extract latent features from HuBERT model."""
        # Configure extraction
        config = ExtractionConfig(
            dataset_csv_path=self.args.dataframe_path,
            ecg_dir=self.input_dir,
            output_dir=self.output_dir,
            batch_size=self.args.batch_size,
            data_slice=(self.args.start_perc, self.args.end_perc),
            save_metadata_csv=self.args.save_csv_for_dumped_features,
            iteration_id=self.args.train_iteration,
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
        
        model = HuBERTECG(model_config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(self.device)
        model.eval()
        
        # Extract features
        extractor = LatentFeatureExtractor(model, self.args.output_layer, config)
        extractor.extract_and_save()
    

    def _get_feature_mode(self) -> str:
        """Determine feature mode from arguments."""
        if self.args.mfcc_only:
            return 'mfcc_only'
        elif self.args.time_freq:
            return 'time_freq'
        else:
            return 'mixed'
    

    def _load_and_slice_dataframe(self) -> pd.DataFrame:
        """Load dataframe and apply slicing."""
        logger.info("Loading dataframe...")
        dataframe = pd.read_csv(self.args.dataframe_path)
        start_idx = int(self.args.start_perc * len(dataframe))
        end_idx = int(self.args.end_perc * len(dataframe)) + 1
        return dataframe.iloc[start_idx:end_idx]


def main():
    """Main entry point for feature extraction."""
    args = create_dumping_parser()
    init_seeds(seed=42)
    pipeline = FeatureExtractionPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
