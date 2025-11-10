import logging
import torch
import torchaudio
import numpy as np
import scipy.stats as stats

from dataclasses import dataclass
from pathlib import Path
from rich.logging import RichHandler
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
   
    def extract(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract 16 time-domain and frequency-domain features.
        
        Args:
            signal: Either 1D array (single signal) or 2D array (batch of signals)
                   Shape: (signal_length,) or (n_signals, signal_length)
       
        Returns:
            Features array of shape (16,) for single signal or (n_signals, 16) for batch
        """
        signal = signal.cpu().numpy()

        # Handle both single and batch inputs
        single_input = signal.ndim == 1
        if single_input:
            signal = signal[np.newaxis, :]  # Add batch dimension
        
        # signal shape: (n_signals, signal_length)
        axis = 1  # Compute along signal dimension
        
        # Time domain features (vectorized across batch)
        signal_min = np.min(signal, axis=axis)
        signal_max = np.max(signal, axis=axis)
        signal_mean = np.mean(signal, axis=axis)
        signal_power = np.mean(signal ** 2, axis=axis)
        signal_rms = np.sqrt(signal_power)
        signal_var = np.var(signal, axis=axis)
        signal_std = np.std(signal, axis=axis)
        signal_peak = np.max(np.abs(signal), axis=axis)
        signal_p2p = np.ptp(signal, axis=axis)
        crest_factor = np.divide(signal_peak, signal_rms, 
                                 out=np.zeros_like(signal_peak), 
                                 where=signal_rms > 0)
        skewness = stats.skew(signal, axis=axis)
        kurtosis = stats.kurtosis(signal, axis=axis)
       
        # Frequency domain features (vectorized)
        power_spectrum = np.abs(fft(signal, axis=axis)) ** 2 / signal.shape[axis]
        spectrum_max = np.max(power_spectrum, axis=axis)
        spectrum_sum = np.sum(power_spectrum, axis=axis)
        spectrum_mean = np.mean(power_spectrum, axis=axis)
        spectrum_var = np.var(power_spectrum, axis=axis)
       
        # Stack features: shape (n_signals, 16)
        features = np.stack([
            signal_min, signal_max, signal_mean, signal_rms, signal_var, signal_std,
            signal_power, signal_peak, signal_p2p, crest_factor, skewness, kurtosis,
            spectrum_max, spectrum_sum, spectrum_mean, spectrum_var
        ], axis=1)
        
        # Return same format as input
        if single_input:
            return features[0]  # Return 1D array for single input
        return features


class MFCCFeatureExtractor(FeatureExtractor):
    """Extract MFCC features with deltas from audio signals."""
   
    def __init__(self, sample_rate: int, device: torch.device = torch.device('cpu')):
        self.sample_rate = sample_rate
        self.device = device
   
    def compute_mfcc_with_deltas_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Compute MFCC features with deltas for a batch of waveforms.
       
        Args:
            waveforms: (batch, time) or (batch, 1, time) tensor
           
        Returns:
            Batched MFCC features of shape (batch, time, 39)
        """
        waveforms = waveforms.to(self.device)
        
        with torch.no_grad():
            if waveforms.dim() == 2:
                # Add channel dimension if needed: (batch, time) -> (batch, 1, time)
                waveforms = waveforms.unsqueeze(1)
           
            batch_size = waveforms.size(0)
            all_features = []
           
            # Process batch through kaldi.mfcc (it handles batches internally)
            for i in range(batch_size):
                waveform = waveforms[i]  # (1, time)
               
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
               
                all_features.append(features)
           
            # Stack into batch: (batch, time, 39)
            return torch.stack(all_features, dim=0)


    def extract(self, signals: torch.Tensor) -> np.ndarray:
        """Extract MFCC features for a batch and return as array."""
        mfccs = self.compute_mfcc_with_deltas_batch(signals)
        batch_size = mfccs.size(0)  # Return as (batch, features) flattened per sample
        return mfccs.reshape(batch_size, -1).cpu().numpy()


class MixedFeatureExtractor(FeatureExtractor):
    """Extract combined time/freq and MFCC features."""
   
    def __init__(self, sample_rate: int, device: torch.device = torch.device('cpu')):
        self.time_freq_extractor = TimeFreqFeatureExtractor()
        self.mfcc_extractor = MFCCFeatureExtractor(sample_rate, device)
   
    def extract(self, signal: torch.Tensor) -> np.ndarray:
        """
        Combine time/freq features with first 13 MFCCs (static coefficients only).
        
        Args:
            signal: Either 1D array (single signal) or 2D array (batch of signals)
                   Shape: (signal_length,) or (n_signals, signal_length)
        
        Returns:
            Features array of shape (29,) for single signal or (n_signals, 29) for batch
            (16 time/freq features + 13 MFCC features)
        """
        # Handle both single and batch inputs
        single_input = signal.ndim == 1 or (hasattr(signal, 'ndim') and signal.ndim == 1)
        
        # Extract time/freq features (returns (16,) or (n_signals, 16))
        time_freq_features = self.time_freq_extractor.extract(signal)
        
        # Extract MFCC features (returns (batch, time, 39))
        mfcc_features = self.mfcc_extractor.extract(signal)

        # Take only first 13 coefficients (static MFCCs) from last dimension
        if mfcc_features.ndim == 3:  # (batch, time, 39)
            mfcc_static = mfcc_features[..., :13]  # (batch, time, 13)
        elif mfcc_features.ndim == 2:  # (time, 39) for single input
            mfcc_static = mfcc_features[:, :13]  # (time, 13)
        
        if single_input:
            # Both should be 1D: (16,) and (13,)
            combined = np.concatenate([time_freq_features, mfcc_static])
        else:
            # Both should be 2D: (n_signals, 16) and (n_signals, 13)
            combined = np.concatenate([time_freq_features, mfcc_static], axis=1)

        return combined


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
