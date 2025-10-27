import concurrent.futures
import logging
import os
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
from typing import Optional, Tuple

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
    """Configuration for different sampling rates."""
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


def _extract_and_save_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    layer_idx: int,
    output_dir: Path,
    expected_hidden_size: int,
    expected_seq_len: int = 93
) -> None:
    """Extract features from model and save to disk."""
    device = next(model.parameters()).device
    
    for batch_idx, (ecgs, ecg_filenames) in enumerate(tqdm(dataloader, desc="Extracting features")):
        ecgs = ecgs.to(device, non_blocking=True)
        
        with torch.no_grad():
            outputs = model(
                ecgs,
                attention_mask=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True
            )
        
        features = outputs['hidden_states'][layer_idx].cpu().numpy()
        
        # Validate feature dimensions
        batch_size, seq_len, hidden_size = features.shape
        assert seq_len == expected_seq_len, (
            f"Unexpected sequence length: {seq_len} (expected {expected_seq_len})"
        )
        assert hidden_size == expected_hidden_size, (
            f"Unexpected hidden size: {hidden_size} (expected {expected_hidden_size})"
        )
        assert batch_size == len(ecg_filenames), (
            f"Batch size mismatch: {batch_size} != {len(ecg_filenames)}"
        )
        
        # Save individual features
        for feature, filename in zip(features, ecg_filenames):
            feature_path = output_dir / f"{Path(filename).stem}.npy"
            np.save(feature_path, feature)

        logger.info(f"Saved batch {batch_idx}: {features.shape}")


def _save_metadata_csv(
    dataframe,
    output_dir: Path,
    layer_idx: int,
    data_slice: Tuple[float, float],
    iteration_id: Optional[int]
) -> None:
    """Save metadata CSV referencing extracted features."""
    start_perc, end_perc = data_slice
    perc_size = int((end_perc - start_perc) * 100)
    iteration_suffix = f"_it{iteration_id}" if iteration_id is not None else ""
    
    csv_path = output_dir / f"latent_{perc_size}perc_layer{layer_idx + 1}{iteration_suffix}.csv"
    dataframe.to_csv(csv_path, index=False)
    logger.info(f"Saved metadata CSV: {csv_path}")


def _create_sliced_dataset(
    dataset_csv_path: Path | str,
    ecg_dir: Path | str,
    downsampling_factor: int,
    data_slice: Tuple[float, float],
) -> 'ECGDataset':
    """Create and slice ECG dataset based on percentage range."""
    dataset = ECGDataset(
        path_to_dataset_csv=dataset_csv_path,
        ecg_dir_path=ecg_dir,
        downsampling_factor=downsampling_factor,
        pretrain=False,
        encode=True,
    )
    
    start_perc, end_perc = data_slice
    start_idx = int(start_perc * len(dataset))
    end_idx = int(end_perc * len(dataset)) + 1
    dataset.ecg_dataframe = dataset.ecg_dataframe.iloc[start_idx:end_idx].copy()
    
    return dataset


def extract_latent_features(
    dataset_csv_path: Path | str,
    ecg_dir: Path | str,
    output_dir: Path | str,
    model: torch.nn.Module,
    layer_idx: int,
    batch_size: int = 32,
    data_slice: Tuple[float, float] = (0.0, 1.0),
    downsampling_factor: int = 5,
    num_workers: int = 0,
    save_metadata_csv: bool = False,
    iteration_id: Optional[int] = None,
) -> None:
    """
    Extract and save latent representations from a HuBERT model.
    
    Args:
        dataset_csv_path: Path to CSV file referencing ECG data
        ecg_dir: Directory containing ECG files
        output_dir: Directory to save extracted features
        model: HuBERT model for encoding ECGs
        layer_idx: Hidden layer index to extract features from (0-indexed)
        batch_size: Batch size for processing
        data_slice: Tuple of (start_fraction, end_fraction) for dataset slicing
        downsampling_factor: Downsampling factor for ECG data
        num_workers: Number of workers for data loading
        save_metadata_csv: Whether to save CSV with feature references
        iteration_id: Optional iteration identifier for metadata CSV naming
    
    Note:
        Features are saved as individual .npy files per ECG sample.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and slice dataset
    dataset = _create_sliced_dataset(
        dataset_csv_path=dataset_csv_path,
        ecg_dir=ecg_dir,
        downsampling_factor=downsampling_factor,
        data_slice=data_slice,
    )
    
    # Save metadata CSV if requested
    if save_metadata_csv:
        _save_metadata_csv(
            dataframe=dataset.ecg_dataframe,
            output_dir=output_dir,
            layer_idx=layer_idx,
            data_slice=data_slice,
            iteration_id=iteration_id,
        )
    
    # Setup dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=dataset.collate,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    
    # Extract and save features
    model.eval()
    _extract_and_save_features(
        model=model,
        dataloader=dataloader,
        layer_idx=layer_idx,
        output_dir=output_dir,
        expected_hidden_size=model.config.hidden_size,
    )


def load_hubert_model(
    checkpoint_path: Path | str,
    device: torch.device,
) -> torch.nn.Module:
    """Load HuBERT model from checkpoint."""
    logger.info("Loading HuBERT model...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_config = checkpoint['model_config']
    model_config.conv_pos_batch_norm = False
    
    model = HuBERTECG(model_config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    return model


def compute_mfcc_with_deltas(
    waveform: torch.Tensor,
    sample_rate: int,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Compute MFCC features with first and second order derivatives (delta, delta-delta).
    
    Args:
        waveform: Input audio tensor of shape (samples,) or (1, samples)
        sample_rate: Sampling rate in Hz
        device: Device for computation
        
    Returns:
        Concatenated MFCC features of shape (time, 39) where 39 = 13 MFCCs * 3
    """
    with torch.no_grad():
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Ensure tensor is on correct device
        waveform = waveform.to(device)
        
        # Compute MFCCs: (time, 13)
        mfccs = torchaudio.compliance.kaldi.mfcc(
            waveform=waveform,
            sample_frequency=sample_rate,
            use_energy=False,
            frame_length=waveform.size(-1) / sample_rate * 1000,
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


def compute_time_freq_features(signal: np.ndarray) -> list:
    """
    Extract 16 time-domain and frequency-domain features from a signal.
    
    Features include:
    - Time domain (12): min, max, mean, RMS, variance, std, power, peak, 
                        peak-to-peak, crest factor, skewness, kurtosis
    - Frequency domain (4): max, sum, mean, variance of power spectrum
    
    Args:
        signal: 1D numpy array of signal values
        
    Returns:
        List of 16 feature values
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


def extract_shard_features(
    shard: np.ndarray,
    feature_mode: str,
    sample_rate: int,
    device: torch.device = torch.device('cpu')
) -> list:
    """
    Extract features from a single audio shard based on the specified mode.
    
    Args:
        shard: Input signal shard (1D numpy array)
        feature_mode: Feature extraction mode
            - 'time_freq': Time and frequency domain features only (16 features)
            - 'mfcc_only': MFCC with deltas (39 features per frame, flattened)
            - 'mixed': Time/freq features + first 13 MFCCs (29 features)
        sample_rate: Sampling rate in Hz
        device: Torch device for MFCC computation
        
    Returns:
        List of feature values
    """
    if feature_mode == 'time_freq':
        return compute_time_freq_features(shard)
    
    # Compute MFCCs (returns single frame for short signals)
    waveform = torch.from_numpy(shard).float()
    mfccs = compute_mfcc_with_deltas(waveform, sample_rate, device)
    mfccs_flat = mfccs.cpu().numpy().flatten().tolist()

    if feature_mode == 'mfcc_only':
        return mfccs_flat
    
    if feature_mode == 'mixed':
        # Combine time/freq features with first 13 MFCCs (static coefficients only)
        time_freq_features = compute_time_freq_features(shard)
        return time_freq_features + mfccs_flat[:13]
    
    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def _trim_lead_data(data: np.ndarray, config: SamplingConfig) -> np.ndarray:
    """
    Trim lead data according to sampling configuration.
    
    Args:
        data: ECG data array of shape (n_leads, n_samples)
        config: Sampling configuration
        
    Returns:
        Trimmed and concatenated 1D array
    """
    trimmed_leads = []
    for i, lead in enumerate(data):
        trim_end = config.trim_end_odd if i % 2 == 1 else config.trim_end_even
        trimmed_leads.append(lead[config.trim_start:-trim_end if trim_end > 0 else None])
    
    return np.concatenate(trimmed_leads)


def _should_skip_extraction(output_path: Path, feature_mode: str) -> bool:
    """Check if feature extraction can be skipped."""
    if not output_path.exists():
        return False
    
    existing_features = np.load(output_path)
    expected_dim = FEATURE_DIMS.get(feature_mode)
    return existing_features.shape[1] == expected_dim


def extract_ecg_features(
    record,
    input_dir: Path,
    output_dir: Path,
    feature_mode: str,
    device: torch.device,
    sample_rate: int,
    max_samples: int = 2500,
    base_sample_rate: int = 500,
) -> Optional[np.ndarray]:
    """
    Extract and save ECG signal features.
    
    Args:
        record: DataFrame record containing filename
        input_dir: Directory containing input ECG data
        output_dir: Directory to save extracted features
        feature_mode: Feature extraction mode ('time_freq', 'mfcc_only', 'mixed')
        device: Torch device for computation
        sample_rate: Target sampling rate (50, 100, or 500 Hz)
        max_samples: Maximum number of samples to process
        base_sample_rate: Original sampling rate of input data
        
    Returns:
        Extracted features array or None if skipped
    """
    filename = record.filename
    input_path = input_dir / filename
    output_path = output_dir / filename
    
    # Skip if features already exist with correct shape
    if _should_skip_extraction(output_path, feature_mode):
        logger.info(f"Skipping {filename}: features already exist")
        return None
    
    # Validate sampling rate
    _validate_sample_rate(sample_rate)
    config = SAMPLING_CONFIGS[sample_rate]
    
    # Load and preprocess data
    data = _load_and_preprocess_ecg(
        input_path=input_path,
        max_samples=max_samples,
        sample_rate=sample_rate,
        base_sample_rate=base_sample_rate,
        filename=filename,
    )
    
    if data is None:
        return None
    
    # Process data into shards and extract features
    features_array = _process_ecg_to_features(
        data=data,
        config=config,
        feature_mode=feature_mode,
        sample_rate=sample_rate,
        device=device,
    )
    
    # Save features
    np.save(output_path, features_array)
    logger.info(f"Saved features to {output_path}")
    
    return features_array




def _validate_sample_rate(sample_rate: int) -> None:
    """Validate that the sample rate is supported."""
    if sample_rate not in SAMPLING_CONFIGS:
        raise ValueError(
            f"Unsupported sample_rate: {sample_rate}. "
            f"Must be one of {list(SAMPLING_CONFIGS.keys())}"
        )


def _load_and_preprocess_ecg(
    input_path: Path,
    max_samples: int,
    sample_rate: int,
    base_sample_rate: int,
    filename: str,
) -> Optional[np.ndarray]:
    """Load and preprocess ECG data with downsampling if needed."""
    data = np.load(input_path)
    data = data[:, :max_samples]
    
    # Handle all-NaN data
    if np.isnan(data).all():
        logger.warning(f"Skipping {filename}: all values are NaN")
        return None
    
    # Downsample if needed
    if sample_rate != base_sample_rate:
        decimation_factor = base_sample_rate // sample_rate
        data = signal.decimate(data, decimation_factor, axis=1)
    
    return data


def _process_ecg_to_features(
    data: np.ndarray,
    config: SamplingConfig,
    feature_mode: str,
    sample_rate: int,
    device: torch.device,
) -> np.ndarray:
    """Process ECG data into feature vectors."""
    # Calculate expected final length
    final_length = data.shape[0] * data.shape[1]
    
    # Trim and concatenate leads
    data_1d = _trim_lead_data(data, config)
    
    # Handle NaN values with mean imputation
    if np.isnan(data_1d).any():
        valid_mean = np.nanmean(data_1d)
        data_1d = np.nan_to_num(data_1d, nan=valid_mean)
    
    # Reshape into shards
    n_shards = final_length // config.compression_factor
    shards = data_1d.reshape(n_shards, config.shard_size)
    
    # Extract features from each shard
    features = [
        extract_shard_features(shard, feature_mode, sample_rate, device)
        for shard in shards
    ]
    
    # Validate output
    _validate_features(features, n_shards, feature_mode)
    
    return np.array(features, dtype=np.float32)


def _validate_features(
    features: list,
    expected_n_shards: int,
    feature_mode: str,
) -> None:
    """Validate extracted features dimensions."""
    expected_dim = FEATURE_DIMS[feature_mode]
    
    assert len(features) == expected_n_shards, (
        f"Expected {expected_n_shards} feature vectors, got {len(features)}"
    )
    assert len(features[0]) == expected_dim, (
        f"Expected {expected_dim} features for mode '{feature_mode}', "
        f"got {len(features[0])}"
    )


def extract_morphological_features(
    dataframe: pd.DataFrame,
    input_dir: Path,
    output_dir: Path,
    feature_mode: str,
    device: torch.device,
    sample_rate: int,
) -> None:
    """Extract morphological features for all records in dataframe."""
    logger.info("Extracting morphological features...")
    
    for record in tqdm(dataframe.itertuples(index=False), total=len(dataframe)):
        extract_ecg_features(
            record=record,
            input_dir=input_dir,
            output_dir=output_dir,
            feature_mode=feature_mode,
            device=device,
            sample_rate=sample_rate,
        )


def main(args):
    """
    Extract ECG features based on training iteration.
    
    Iteration 1: Extract morphological features
    Iteration 2+: Extract latent features from HuBERT model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_seeds(seed=42)
    
    input_dir = Path(args.in_dir)
    output_dir = Path(args.dest_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine feature mode
    if args.mfcc_only:
        feature_mode = 'mfcc_only'
    elif args.time_freq:
        feature_mode = 'time_freq'
    else:
        feature_mode = 'mixed'
    
    # Load and slice dataframe
    logger.info("Loading dataframe...")
    dataframe = pd.read_csv(args.dataframe_path)
    start_idx = int(args.start_perc * len(dataframe))
    end_idx = int(args.end_perc * len(dataframe)) + 1
    dataframe = dataframe.iloc[start_idx:end_idx]

    # Extract features based on iteration
    if args.train_iteration == 1:
        extract_morphological_features(
            dataframe=dataframe,
            input_dir=input_dir,
            output_dir=output_dir,
            feature_mode=feature_mode,
            device=device,
            sample_rate=args.samp_rate,
        )
    else:
        model = load_hubert_model(args.hubert_path, device)
        logger.info(
            f"Extracting latent features from layer {args.output_layer + 1} "
            f"of HuBERT encoder..."
        )
        extract_latent_features(
            dataset_csv_path=args.dataframe_path,
            ecg_dir=input_dir,
            output_dir=output_dir,
            model=model,
            layer_idx=args.output_layer,
            data_slice=(args.start_perc, args.end_perc),
            iteration_id=args.train_iteration,
            batch_size=args.batch_size,
            save_metadata_csv=args.save_csv_for_dumped_features,
        )
    
    logger.info("Feature extraction complete.")


if __name__ == "__main__":
    args = create_dumping_parser()
    main(args)
