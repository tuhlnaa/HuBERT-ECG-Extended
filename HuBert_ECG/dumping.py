import argparse
import concurrent.futures
import os
import random
import torch
import torchaudio

import numpy as np
import pandas as pd
import scipy.stats as stats

from dataset import ECGDataset
from loguru import logger
from scipy import signal
from scipy.fft import fft
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import custom modules
from hubert_ecg import HuBERTECG, HuBERTECGConfig
from config import create_dumping_parser, init_seeds

SHARD_SIZE_500 = 322
SHARD_SIZE_100 = 64
SHARD_SIZE_50 = 32

CNN_COMPRESSION_FACTOR_500 = 320
CNN_COMPRESSION_FACTOR_100 = 64
CNN_COMPRESSION_FACTOR_50 = 32


def compute_mfcc_features_and_derivatives(x : torch.Tensor, samp_rate : int):
    ''' Compute MFCC features and their first and second derivatives from a signal.'''
    
    with torch.no_grad():
        x = x.view(1, -1)

        mfccs = torchaudio.compliance.kaldi.mfcc(
            waveform=x,
            sample_frequency=samp_rate,
            use_energy=False,
            frame_length= x.size(-1) / samp_rate * 1000,
            frame_shift=100
        )  # (time, freq)
        mfccs = mfccs.transpose(0, 1)  # (freq, time)
        deltas = torchaudio.functional.compute_deltas(mfccs)
        ddeltas = torchaudio.functional.compute_deltas(deltas)
        concat = torch.cat([mfccs, deltas, ddeltas], dim=0)
        concat = concat.transpose(0, 1).contiguous()  # (freq, time)
        return concat # (1, 39) torch.Tensor


def get_signal_features(signal):
    '''Extracts 17 features from a signal considering both time and frequency domain'''

    ## TIME DOMAIN ##
    Min = (np.min(signal))
    Max = (np.max(signal))
    Mean = (np.mean(signal))
    Power = (np.mean(signal**2))
    Rms = (np.sqrt(Power))
    Var = (np.var(signal))
    Std = (np.std(signal))
    Peak = (np.max(np.abs(signal)))
    P2p = (np.ptp(signal))
    CrestFactor = (Peak/Rms)
    Skew = (stats.skew(signal))
    Kurtosis = (stats.kurtosis(signal))
    
    ## FREQ DOMAIN ##
    ft = fft(signal)
    S = np.abs(ft**2)/len(signal) 
    Max_f = (np.max(S))
    Sum_f = (np.sum(S))
    Mean_f = (np.mean(S))
    Var_f = (np.var(S))     
    
    features = [Min,Max,Mean,Rms,Var,Std,Power,Peak,P2p,CrestFactor,Skew,Kurtosis,Max_f,Sum_f,Mean_f,Var_f]
        
    return features


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


def dump_ecg_features(record, in_dir, dest_dir, mfcc_only, time_freq, device, samp_rate):
    '''
    Save on disk the features of the ECG signal concatenated in a single vector.

    Args:
        - record from the dataframe apply() function was called on
        - in_dir is the path to th input directory where the data pointed by the record is
        - dest_dir is the path to the output directory where the features will be saved
        - mfcc_only is a flag used when one wants to compute the 39 MFCCs. It's mutex with time_freq
        - time_freq is a flag used when one wants to compute the 16 time-frequency features. it's mutex with mfcc_only
        - samp_rate is an integer indicating the desidered sampling rate. Used when some or all MFCCs are to compute
    '''
    filename = record.filename
    path = os.path.join(dest_dir, filename)
    
    if samp_rate == 500:
        shard_length = SHARD_SIZE_500
        cnn_compression_factor = CNN_COMPRESSION_FACTOR_500
    elif samp_rate == 100:
        shard_length = SHARD_SIZE_100
        cnn_compression_factor = CNN_COMPRESSION_FACTOR_100
    else: # 50 Hz
        shard_length = SHARD_SIZE_50
        cnn_compression_factor = CNN_COMPRESSION_FACTOR_50
    
    if np.load(path).shape[0] != 93: #if features do not exist then calculate them, skip otherwise
        
        data = np.load(os.path.join(in_dir, filename))
        data = data[:, :2500]

        if np.isnan(data).all():
            return 

        data = signal.decimate(data, int(500/samp_rate))

        final_length = data.shape[0] * data.shape[1]

        shortened_data = []
        if samp_rate == 500:
            for i in range(len(data)):
                if i % 2 == 0:
                    shortened_data.append(data[i, 2:-2])
                else:
                    shortened_data.append(data[i, 2:-3])
        elif samp_rate == 100:
            for i in range(len(data)):
                shortened_data.append(data[i, 2:-2])             
        else: # 50 Hz
            for i in range(len(data)):
                shortened_data.append(data[i, 1:-1])

        data = np.concatenate(shortened_data)
        
        mask = np.isnan(data)
        data = np.where(mask, data[~mask].mean(), data)
        
        shards = data.reshape(-1, shard_length)
        assert shards.shape[0] == final_length // cnn_compression_factor and shards.shape[1] == shard_length, f"{shards.shape}"


        ####
        
        features = []
        for shard in shards:
            if time_freq:
                features.append(get_signal_features(shard))
            else:                
                mfccs = compute_mfcc_features_and_derivatives(torch.from_numpy(shard).to(device), samp_rate)
                mfccs = mfccs.cpu().numpy().tolist()[0] # [39 coeffs]
                if mfcc_only:
                    features.append(mfccs)
                else: # mixed
                    signal_features = get_signal_features(shard)
                    features.append(signal_features + mfccs[:13]) #(93, 30)
                
        ####
        
        assert len(features) == final_length // cnn_compression_factor, f"len(features)={len(features)}"
        if time_freq:
            assert len(features[0]) == 16, f"Time freq features, detected length {len(features[0])}"
        elif mfcc_only:
            assert len(features[0]) == 39, f"MFCC only features, detected length {len(features[0])}"
        else:
            assert len(features[0]) == 29, f"Mixed features, detected length {len(features[0])}"

        features = np.array(features, dtype=np.float32)
        np.save(os.path.join(dest_dir, filename[:-4]), features)
    else:
        logger.info(f"Skipping {filename} because features already exist")


def main(args):
    '''
    Function called with arguments passed through shell and used to dump both morphological and latent features.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_seeds(seed=42)
    
    if args.train_iteration == 1:
        logger.info("Loading dataframe...")
        dataframe = pd.read_csv(args.dataframe_path)
        dataframe = dataframe.iloc[int(args.start_perc * len(dataframe)) : int(args.end_perc * len(dataframe))+1]
        logger.info("Dumping morphological features...")
        dataframe.apply(dump_ecg_features, axis=1, args=(args.in_dir, args.dest_dir, args.mfcc_only, args.time_freq, device, args.samp_rate,))
    else:
        logger.info("Loading HuBERT model to get latent features from...")
        checkpoint = torch.load(args.hubert_path, map_location='cpu')
        hubert = HuBERTECG(checkpoint['model_config'])
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
