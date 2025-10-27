import logging
import torch
import torchaudio

import numpy as np
import pandas as pd
import scipy.stats as stats

from dataclasses import dataclass
from pathlib import Path
from rich.logging import RichHandler
from scipy import signal
from scipy.fft import fft
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Optional, Tuple

# Import custom modules
from config import create_dumping_parser, init_seeds
from dataset import ECGDataset
from hubert_ecg import HuBERTECG, HuBERTECGConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", 
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class SamplingConfig:
    """Configuration for ECG signal sampling and processing."""
    shard_size: int
    compression_factor: int
    trim_start: int
    trim_end_even: int
    trim_end_odd: int


# Configuration mapping for different sampling rates
SAMPLING_CONFIGS = {
    500: SamplingConfig(
        shard_size=322, 
        compression_factor=320,
        trim_start=2, 
        trim_end_even=2, 
        trim_end_odd=3
    ),
    100: SamplingConfig(
        shard_size=64, 
        compression_factor=64,
        trim_start=2, 
        trim_end_even=2, 
        trim_end_odd=2
    ),
    50: SamplingConfig(
        shard_size=32, 
        compression_factor=32,
        trim_start=1, 
        trim_end_even=1, 
        trim_end_odd=1
    ),
}

# Feature dimensions for validation
FEATURE_DIMS = {
    'time_freq': 16,
    'mfcc_only': 39,
    'mixed': 29,
}


class FeatureExtractor:
    """Base class for feature extraction strategies."""
    
    def extract(self, signal: np.ndarray) -> List[float]:
        raise NotImplementedError


class TimeFreqFeatureExtractor(FeatureExtractor):
    """Extract time-domain and frequency-domain features from signals."""
    
    def extract(self, signal: np.ndarray) -> List[float]:
        """
        Extract 16 time-domain and frequency-domain features.
        
        Features include:
        - Time domain (12): min, max, mean, RMS, variance, std, power, peak, 
                            peak-to-peak, crest factor, skewness, kurtosis
        - Frequency domain (4): max, sum, mean, variance of power spectrum
        """
        # Time domain features
        signal_min = np.min(signal)
        signal_max = np.max(signal)
        signal_mean = np.mean(signal)
        signal_power = np.mean(signal ** 2)
        signal_rms = np.sqrt(signal_power)
        signal_var = np.var(signal)
        signal_std = np.std(signal)
        signal_peak = np.max(np.abs(signal))
        signal_p2p = np.ptp(signal)
        crest_factor = signal_peak / signal_rms if signal_rms > 0 else 0.0
        skewness = stats.skew(signal)
        kurtosis = stats.kurtosis(signal)
        
        # Frequency domain features
        power_spectrum = np.abs(fft(signal)) ** 2 / len(signal)
        spectrum_max = np.max(power_spectrum)
        spectrum_sum = np.sum(power_spectrum)
        spectrum_mean = np.mean(power_spectrum)
        spectrum_var = np.var(power_spectrum)
        
        return [
            signal_min, signal_max, signal_mean, signal_rms, signal_var, signal_std,
            signal_power, signal_peak, signal_p2p, crest_factor, skewness, kurtosis,
            spectrum_max, spectrum_sum, spectrum_mean, spectrum_var
        ]


class MFCCFeatureExtractor(FeatureExtractor):
    """Extract MFCC features with deltas from audio signals."""
    
    def __init__(self, sample_rate: int, device: torch.device = torch.device('cpu')):
        self.sample_rate = sample_rate
        self.device = device
    
    def compute_mfcc_with_deltas(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute MFCC features with first and second order derivatives.
        
        Returns:
            Concatenated MFCC features of shape (time, 39) where 39 = 13 MFCCs * 3
        """
        with torch.no_grad():
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            waveform = waveform.to(self.device)
            
            # Compute MFCCs: (time, 13)
            mfccs = torchaudio.compliance.kaldi.mfcc(
                waveform=waveform,
                sample_frequency=self.sample_rate,
                use_energy=False,
                frame_length=waveform.size(-1) / self.sample_rate * 1000,
                frame_shift=100
            )
            
            # Transpose for delta computation: (13, time)
            mfccs_t = mfccs.transpose(0, 1)
            deltas = torchaudio.functional.compute_deltas(mfccs_t)
            delta_deltas = torchaudio.functional.compute_deltas(deltas)
            
            # Concatenate and transpose back: (time, 39)
            features = torch.cat([mfccs_t, deltas, delta_deltas], dim=0)
            features = features.transpose(0, 1).contiguous()
            
            return features
    
    def extract(self, signal: np.ndarray) -> List[float]:
        """Extract MFCC features and return as flattened list."""
        waveform = torch.from_numpy(signal).float()
        mfccs = self.compute_mfcc_with_deltas(waveform)
        return mfccs.cpu().numpy().flatten().tolist()


class MixedFeatureExtractor(FeatureExtractor):
    """Extract combined time/freq and MFCC features."""
    
    def __init__(self, sample_rate: int, device: torch.device = torch.device('cpu')):
        self.time_freq_extractor = TimeFreqFeatureExtractor()
        self.mfcc_extractor = MFCCFeatureExtractor(sample_rate, device)
    
    def extract(self, signal: np.ndarray) -> List[float]:
        """Combine time/freq features with first 13 MFCCs (static coefficients only)."""
        time_freq_features = self.time_freq_extractor.extract(signal)
        mfcc_features = self.mfcc_extractor.extract(signal)
        return time_freq_features + mfcc_features[:13]


class FeatureExtractorFactory:
    """Factory for creating feature extractors based on mode."""
    
    FEATURE_DIMS = {
        'time_freq': 16,
        'mfcc_only': 39,
        'mixed': 29,
    }
    
    @staticmethod
    def create(feature_mode: str, sample_rate: int, 
               device: torch.device = torch.device('cpu')) -> FeatureExtractor:
        """Create appropriate feature extractor based on mode."""
        if feature_mode == 'time_freq':
            return TimeFreqFeatureExtractor()
        elif feature_mode == 'mfcc_only':
            return MFCCFeatureExtractor(sample_rate, device)
        elif feature_mode == 'mixed':
            return MixedFeatureExtractor(sample_rate, device)
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")
    
    @classmethod
    def get_feature_dim(cls, feature_mode: str) -> int:
        """Get expected feature dimension for a given mode."""
        return cls.FEATURE_DIMS[feature_mode]


class ECGDataProcessor:
    """Process ECG data with configurable sampling and trimming."""
    
    def __init__(self, config: SamplingConfig, sample_rate: int, 
                 base_sample_rate: int = 500):
        self.config = config
        self.sample_rate = sample_rate
        self.base_sample_rate = base_sample_rate
    

    def load_and_preprocess(self, input_path: Path, max_samples: int, 
                           filename: str) -> Optional[np.ndarray]:
        """Load and preprocess ECG data with downsampling if needed."""
        data = np.load(input_path)
        data = data[:, :max_samples]
        
        # Handle all-NaN data
        if np.isnan(data).all():
            logger.warning(f"Skipping {filename}: all values are NaN")
            return None
        
        # Downsample if needed
        if self.sample_rate != self.base_sample_rate:
            decimation_factor = self.base_sample_rate // self.sample_rate
            data = signal.decimate(data, decimation_factor, axis=1)
        
        return data
    

    def trim_lead_data(self, data: np.ndarray) -> np.ndarray:
        """Trim lead data according to sampling configuration."""
        trimmed_leads = []
        for i, lead in enumerate(data):
            trim_end = self.config.trim_end_odd if i % 2 == 1 else self.config.trim_end_even
            trimmed_leads.append(
                lead[self.config.trim_start:-trim_end if trim_end > 0 else None]
            )
        return np.concatenate(trimmed_leads)
    

    def process_to_shards(self, data: np.ndarray) -> np.ndarray:
        """Process ECG data into shards for feature extraction."""
        # Trim and concatenate leads
        data_1d = self.trim_lead_data(data)
        
        # Handle NaN values with mean imputation
        if np.isnan(data_1d).any():
            valid_mean = np.nanmean(data_1d)
            data_1d = np.nan_to_num(data_1d, nan=valid_mean)
        
        # Reshape into shards
        final_length = data.shape[0] * data.shape[1]
        n_shards = final_length // self.config.compression_factor
        shards = data_1d.reshape(n_shards, self.config.shard_size)
        
        return shards


class ECGFeatureExtractor:
    """Main class for extracting features from ECG records."""
    
    def __init__(self, sampling_configs: dict, device: torch.device = torch.device('cpu')):
        self.sampling_configs = sampling_configs
        self.device = device
    

    def should_skip_extraction(self, output_path: Path, feature_mode: str) -> bool:
        """Check if feature extraction can be skipped."""
        if not output_path.exists():
            return False
        
        existing_features = np.load(output_path)
        expected_dim = FeatureExtractorFactory.get_feature_dim(feature_mode)
        return existing_features.shape[1] == expected_dim
    

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
    

    def extract_features(self, record, input_dir: Path, output_dir: Path,
                        feature_mode: str, sample_rate: int, max_samples: int = 2500,
                        base_sample_rate: int = 500) -> Optional[np.ndarray]:
        """
        Extract and save ECG signal features.
        
        Returns:
            Extracted features array or None if skipped
        """
        filename = record.filename
        input_path = input_dir / filename
        output_path = output_dir / filename
        
        # Skip if features already exist with correct shape
        if self.should_skip_extraction(output_path, feature_mode):
            logger.info(f"Skipping {filename}: features already exist")
            return None
        
        # Validate sampling rate
        if sample_rate not in self.sampling_configs:
            raise ValueError(
                f"Unsupported sample_rate: {sample_rate}. "
                f"Must be one of {list(self.sampling_configs.keys())}"
            )
        
        config = self.sampling_configs[sample_rate]
        
        # Load and preprocess data
        processor = ECGDataProcessor(config, sample_rate, base_sample_rate)
        data = processor.load_and_preprocess(input_path, max_samples, filename)
        
        if data is None:
            return None
        
        # Process data into shards
        shards = processor.process_to_shards(data)
        
        # Extract features from each shard
        feature_extractor = FeatureExtractorFactory.create(
            feature_mode, sample_rate, self.device
        )
        features = [feature_extractor.extract(shard) for shard in shards]
        
        # Validate output
        expected_n_shards = (data.shape[0] * data.shape[1]) // config.compression_factor
        self.validate_features(features, expected_n_shards, feature_mode)
        
        # Save features
        features_array = np.array(features, dtype=np.float32)
        np.save(output_path, features_array)
        logger.info(f"Saved features to {output_path}")
        
        return features_array
    

    def extract_batch(self, dataframe: pd.DataFrame, input_dir: Path, 
                     output_dir: Path, feature_mode: str, sample_rate: int) -> None:
        """Extract morphological features for all records in dataframe."""
        logger.info("Extracting morphological features...")
        
        for record in tqdm(dataframe.itertuples(index=False), total=len(dataframe)):
            self.extract_features(
                record=record,
                input_dir=input_dir,
                output_dir=output_dir,
                feature_mode=feature_mode,
                sample_rate=sample_rate,
            )


@dataclass
class ExtractionConfig:
    """Configuration for feature extraction."""
    dataset_csv_path: Path
    ecg_dir: Path
    output_dir: Path
    batch_size: int = 32
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
        
        init_seeds(seed=42)
    

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
        
        logger.info("Feature extraction complete.")
    

    def _extract_morphological_features(self) -> None:
        """Extract morphological features using ECGFeatureExtractor."""
        logger.info("Extracting morphological features...")
        
        # Determine feature mode
        feature_mode = self._get_feature_mode()
        
        # Load and slice dataframe
        dataframe = self._load_and_slice_dataframe()
        
        # Extract features
        extractor = ECGFeatureExtractor(SAMPLING_CONFIGS, self.device)
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
        #model = HuBERTModelLoader.load(self.args.hubert_path, self.device)
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


def main(args):
    """Main entry point for feature extraction."""
    pipeline = FeatureExtractionPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    args = create_dumping_parser()
    main(args)