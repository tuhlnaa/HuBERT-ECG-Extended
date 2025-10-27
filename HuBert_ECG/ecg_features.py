import logging
import torch
import torchaudio

import numpy as np
import pandas as pd
import scipy.stats as stats

from dataclasses import dataclass
from pathlib import Path
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
from scipy import signal
from scipy.fft import fft
from typing import Dict, List, Optional, Tuple


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


@dataclass
class FeatureConfig:
    """Configuration for feature extraction modes."""
    name: str
    dimension: int


# Consolidated configuration
class Config:
    """Central configuration for sampling rates and feature modes."""
    
    # Configuration mapping for different sampling rates
    SAMPLING = {
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
    
    # Feature dimensions for each extraction mode
    FEATURES = {
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
    
    @staticmethod
    def get_feature_dim(feature_mode: str) -> int:
        """Get expected feature dimension for a given mode."""
        return Config.FEATURES[feature_mode]


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
    
    def __init__(self, device: torch.device = torch.device('cpu')):
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
            return None
        
        # Validate sampling rate
        if sample_rate not in Config.SAMPLING:
            error_msg = (
                f"Unsupported sample_rate: {sample_rate}. "
                f"Must be one of {list(Config.SAMPLING.keys())}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        config = Config.SAMPLING[sample_rate]
        
        # Load and preprocess data
        processor = ECGDataProcessor(config, sample_rate, base_sample_rate)
        data = processor.load_and_preprocess(input_path, max_samples, filename)
        
        if data is None:
            logger.warning(f"Failed to load and preprocess data for {filename}")
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
        
        return features_array
        

    def extract_batch(self, dataframe: pd.DataFrame, input_dir: Path, 
                    output_dir: Path, feature_mode: str, sample_rate: int) -> Dict[str, int]:
        """
        Extract morphological features for all records in dataframe.
        
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
        
        processed_count = 0
        failed_count = 0
        skipped_count = 0
        failed_records = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
        ) as progress:
            
            task_id = progress.add_task("[green]Extracting features", total=total_records)
            
            for record in dataframe.itertuples(index=False):
                try:
                    result = self.extract_features(
                        record=record,
                        input_dir=input_dir,
                        output_dir=output_dir,
                        feature_mode=feature_mode,
                        sample_rate=sample_rate,
                    )
                    
                    if result is None:
                        skipped_count += 1
                    else:
                        processed_count += 1
                        
                except Exception as e:
                    failed_count += 1
                    error_msg = f"Failed to extract features for {record.filename}: {str(e)}"
                    failed_records.append((record.filename, error_msg))
                    logger.error(error_msg)
                
                progress.update(task_id, advance=1)
        
        self._log_failed_records(failed_records, failed_count)
        
        stats = {
            "processed": processed_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "total": total_records,
        }
        
        logger.info(f"Feature extraction complete: {stats}")
        return stats


    def _log_failed_records(self, failed_records: List[Tuple[str, str]], 
                           failed_count: int) -> None:
        """Log summary of failed records."""
        if not failed_records:
            return
        
        logger.error(f"{failed_count} records failed feature extraction:")
        for filename, error_msg in failed_records[:3]:
            logger.error(f"  {filename}: {error_msg}")
        
        if len(failed_records) > 3:
            logger.error(f"  ... and {len(failed_records) - 3} more failures")
