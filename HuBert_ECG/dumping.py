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
from scipy.fft import fft, rfft
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

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
    500: SamplingConfig(shard_size=322, compression_factor=320, 
                        trim_start=2, trim_end_even=2, trim_end_odd=3),
    100: SamplingConfig(shard_size=64, compression_factor=64,
                        trim_start=2, trim_end_even=2, trim_end_odd=2),
    50: SamplingConfig(shard_size=32, compression_factor=32,
                       trim_start=1, trim_end_even=1, trim_end_odd=1),
}

# Feature dimensions for validation
FEATURE_DIMS = {
    'time_freq': 16,
    'mfcc_only': 39,
    'mixed': 29,
}


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


def dump_latent_features(path_to_dataset_csv, in_dir, dest_dir, start_perc, end_perc, hubert, output_layer, iteration, batch_size, save_csv):
    '''
    Saves on disk computed latent representation once extracted from `hubert`'s `output_layer`.
    Args:
    - path_to_dataset_csv: path to the csv files referencing the ECGs
    - in_dir: where the ECGs are
    - dest_dir: where to save the computed representations
    - start_perc and end_perc indicate the starting and ending point of the csv file of which latents are to compute
    Example: data_set.ecg_dataframe.iloc[int(start_perc * len(data_set)) : int(end_perc * len(data_set))+1]
    - hubert: a hubert model used to encode raw ECG into representations
    - output_layer: the encoding layer from which latents are to be extracted
    - iteration: iteration id used when saving the csv file referencing the saved representations
    - batch_size: the batch_size to use when feeding ECGs into hubert. 
    - save_csv: whether to save a csv file referencing dumped features.
    '''
        
    data_set = ECGDataset(
        path_to_dataset_csv = path_to_dataset_csv,
        ecg_dir_path = in_dir,
        downsampling_factor=5,
        pretrain = False,
        encode = True
    )
    
    # cutting dataframe to the desired percentage
    data_set.ecg_dataframe = data_set.ecg_dataframe.iloc[int(start_perc * len(data_set)) : int(end_perc * len(data_set))+1]

    if save_csv:
        data_set.ecg_dataframe.to_csv(f"latent_{int((end_perc-start_perc)*100)}_perc_encoder_{output_layer+1}_it{iteration}.csv", index=False)
        logger.info("Saved csv file containing references to dumped latents")
    
    dataloader = DataLoader(
        data_set,
        batch_size=batch_size,
        num_workers=5,
        collate_fn=data_set.collate,
        drop_last=False
    )
    
    hubert.eval()
    
    for i, (ecgs, ecg_filenames) in enumerate(tqdm(dataloader, total=len(dataloader))):
        
        ecgs = ecgs.to(hubert.device)
        
        with torch.no_grad():
            out_encoder = hubert(ecgs, attention_mask=None, output_attentions=False, output_hidden_states=True, return_dict=True)
            
        features = out_encoder['hidden_states'][output_layer]
        
        assert features.size(1) == 93 and features.size(2) == hubert.config.hidden_size, f"{features.shape} , {ecg_filenames}"
        assert features.size(0) == len(ecg_filenames), f"{features.size(0)} != {len(ecg_filenames)}"
        
        features = features.cpu().numpy() # (B, n_tokens, D)
        
        # # save batched features in a single file
        # path = os.path.join(dest_dir, f"batch_{i}.npy")
        # block_mapping[path] = ecg_filenames
        # np.save(path[:-4], features)
        
        ecg_paths = [os.path.join(dest_dir, ecg_filename[:-4]) for ecg_filename in ecg_filenames] # new list for every batch
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(np.save, ecg_paths, features)
        
        logger.info(f"Saved batch of features with shape {features.shape}") 


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
    if output_path.exists():
        existing_features = np.load(output_path)
        expected_dim = FEATURE_DIMS.get(feature_mode)
        if existing_features.shape[1] == expected_dim:
            logger.info(f"Skipping {filename}: features already exist")
            return None
    
    # Validate sampling rate
    if sample_rate not in SAMPLING_CONFIGS:
        raise ValueError(f"Unsupported sample_rate: {sample_rate}. Must be one of {list(SAMPLING_CONFIGS.keys())}")
    
    config = SAMPLING_CONFIGS[sample_rate]
    
    # Load and preprocess data
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
    expected_dim = FEATURE_DIMS[feature_mode]
    assert len(features) == n_shards, \
        f"Expected {n_shards} feature vectors, got {len(features)}"
    assert len(features[0]) == expected_dim, \
        f"Expected {expected_dim} features for mode '{feature_mode}', got {len(features[0])}"
    
    # Save features
    features_array = np.array(features, dtype=np.float32)
    np.save(output_path, features_array)
    logger.info(f"Saved features to {output_path}")
    
    return features_array


def main(args):
    '''
    Function called with arguments passed through shell and used to dump both morphological and latent features.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_seeds(seed=42)
    
    in_dir = Path(args.in_dir)
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if args.mfcc_only:
        feature_mode = 'mfcc_only'  # output shape: (93, 39)
    elif args.time_freq:
        feature_mode = 'time_freq'  # output shape: (93, 16)
    else:
        feature_mode = 'mixed'  # output shape: (93, 29)

    if args.train_iteration == 1:
        logger.info("Loading dataframe...")
        dataframe = pd.read_csv(args.dataframe_path)
        dataframe = dataframe.iloc[int(args.start_perc * len(dataframe)) : int(args.end_perc * len(dataframe))+1]
        logger.info("Dumping morphological features...")
        dataframe.apply(extract_ecg_features, axis=1, args=(in_dir, dest_dir, feature_mode, device, args.samp_rate))
    else:
        logger.info("Loading HuBERT model to get latent features from...")
        checkpoint = torch.load(args.hubert_path, map_location='cpu', weights_only=False)
        model_config = checkpoint['model_config']
        model_config.conv_pos_batch_norm = False

        hubert = HuBERTECG(model_config)
        hubert.load_state_dict(checkpoint['model_state_dict'], strict=False)
        hubert = hubert.to(device)
        hubert.eval()
        #dataframe.apply(dump_ecg_features_from_hubert, axis=1, args=(args.in_dir, hubert, 5, args.dest_dir, ))
        logger.info(f"Dumping latent features from {args.output_layer + 1}th layer of HuBERT's encoder...")
        dump_latent_features(args.dataframe_path, args.in_dir, args.dest_dir, args.start_perc, args.end_perc, hubert, args.output_layer, args.train_iteration, batch_size=args.batch_size, save_csv=args.save_csv_for_dumped_features)
    
    logger.info("Features dumped.")


if __name__ == "__main__":
    args = create_dumping_parser()
    main(args)
