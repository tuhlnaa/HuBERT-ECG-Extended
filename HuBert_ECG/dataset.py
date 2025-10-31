import logging
import numpy as np
import pandas as pd
from typing import Tuple, Any
import torch
import os
import joblib
from torch.utils.data import Dataset
import neurokit2 as nk
from scipy import signal
from torch.utils.data import DataLoader
from rich.logging import RichHandler

SAMPLES_IN_5_SECONDS_AT_500HZ = 2500
SAMPLES_IN_10_SECONDS_AT_500HZ = 5000



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", 
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


def create_dataloader(
    csv_path: str,
    ecg_dir: str,
    batch_size: int,
    label_start_idx: int = 3,
    downsample_factor: int = None,
    random_crop: bool = False,
    shuffle: bool = True,
    is_pretrain: bool = False,
    kmeans_path: str = None,
    features_path: str = None,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader for ECG dataset.
    
    Args:
        csv_path: Path to dataset CSV file
        ecg_dir: Directory containing ECG data
        batch_size: Batch size for DataLoader
        label_start_idx: Starting index of labels in CSV
        downsample_factor: Factor for downsampling ECG signals
        random_crop: Whether to apply random 5s crop augmentation
        shuffle: Whether to shuffle data
        is_pretrain: Whether this is for pretraining mode
        kmeans_path: Path to text file containing paths to K-means models
            for ensemble label assignment (used in pretraining)
        features_path: Directory path to dumped features extracted from shards
            or transformer mid-layers (used in pretraining)
        drop_last: Whether to drop the last incomplete batch

    Returns:
        Configured DataLoader instance
    """
    dataset = ECGDataset(
        path_to_dataset_csv=csv_path,
        ecg_dir_path=ecg_dir,
        label_start_index=label_start_idx,
        downsampling_factor=downsample_factor,
        pretrain=is_pretrain,
        random_crop=random_crop,
        kmeans_path=kmeans_path,
        features_path=features_path
    )

    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty! No images found. Please check the paths and file formats.")
    
    data_loader = DataLoader(
        dataset,
        collate_fn=dataset.collate,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=drop_last,
    )
    logger.info(f"Dataset samples: {len(dataset)}, DataLoader batches: {len(data_loader)}")

    return data_loader


def validate_vocab_sizes(args, dataset):
    """Validate vocabulary sizes match k-means cluster counts."""
    assert len(args.vocab_sizes) == dataset.ensamble_length, (
        f"Number of vocab_sizes ({len(args.vocab_sizes)}) must match "
        f"number of tasks ({dataset.ensamble_length})"
    )
    
    for vocab_size, kmeans in zip(args.vocab_sizes, dataset.ensamble_kmeans):
        n_clusters = kmeans.cluster_centers_.shape[0]
        assert vocab_size == n_clusters, (
            f"vocab_size ({vocab_size}) must match number of k-means "
            f"clusters ({n_clusters})"
        )


class ECGDataset(Dataset):
    def __init__(
        self,
        path_to_dataset_csv : str,
        ecg_dir_path : str,
        downsampling_factor : int = None,
        features_path : str = None,
        kmeans_path : str = None,
        label_start_index : int = 3,
        pretrain : bool = True,
        encode : bool = False,
        beat_based_attention_mask : bool = False,
        random_crop : bool = False,
        return_full_length : bool = False,
        ):
        """Dataset for loading and processing ECG signals for pretraining and finetuning.

        Args:
            path_to_dataset_csv: Path to dataset CSV file. For multilabel classification,
                the CSV should contain binary columns for each label. For multiclass classification,
                a single column with integer values in range [0, n_classes-1] is expected.
            ecg_dir_path: Directory path containing raw ECG files.
            downsampling_factor: Downsampling factor to apply to ECG signals.
            features_path: Directory path to dumped features extracted from shards
                or transformer mid-layers. Used only when `pretrain=True`.
            kmeans_path: Path to text file containing paths to K-means models
                used for assigning ensemble labels to features. Used when `pretrain=True`.
            label_start_index: Column index in CSV where labels begin.
                Used when `pretrain=False` and `encode=False`.
            pretrain: Whether pretraining mode is active.
            encode: Whether to enable encoding mode for faster feature dumping.
            beat_based_attention_mask: Whether to calculate beat-based attention
                mask focusing on P wave, QRS complex, and T wave.
            random_crop: Whether to randomly crop ECG signals. Should only be
                used during finetuning and testing to avoid misalignments between signals and
                features.
            return_full_length: Whether to return full 10-second ECG signals
                instead of 5-second segments.

        Note:
            The `__getitem__` method returns different outputs based on mode:
            
            * **Pretrain mode** (`pretrain=True`):
                - ecg_data: Float tensor of shape (12 * length / downsampling_factor,)
                - attention_mask: Long tensor of shape (12 * length / downsampling_factor,)
                - labels: Long tensor of shape (ensemble_length, n_tokens)
            
            * **Encode mode** (`encode=True`, `pretrain=False`):
                - ecg_data: Float tensor of shape (12 * length / downsampling_factor,)
                - ecg_filename: String indicating the ECG filename
            
            * **Finetuning mode** (`pretrain=False`, `encode=False`):
                - ecg_data: Float tensor of shape (12 * length / downsampling_factor,)
                - attention_mask: Long tensor of shape (12 * length / downsampling_factor,)
                - labels: Float tensor of shape (n_classes,) for multilabel or Long tensor for multiclass
            
            where length = 5000 if `return_full_length=True` else 2500.
        """
        
        logger.info(f"Loading dataset from {path_to_dataset_csv}...")

        self.ecg_dataframe = pd.read_csv(path_to_dataset_csv, dtype={'filename': str})
        self.ecg_dir_path = ecg_dir_path
        self.downsampling_factor = downsampling_factor
        self.pretrain = pretrain
        self.encode = encode
        self.beat_based_attention_mask = beat_based_attention_mask
        self.random_crop = random_crop
        self.return_full_length = return_full_length

        if pretrain:
            with open(kmeans_path, 'r') as f:
                kmeans_paths = f.readlines()
                
            # filter out commented lines
            kmeans_paths = [path for path in kmeans_paths if not path.startswith("#")]
                
            self.ensamble_length = len(kmeans_paths)
            self.ensamble_kmeans = [joblib.load(path.strip()) for path in kmeans_paths]
            self.features_path = features_path
        elif not encode:
            self.diagnoses_cols = self.ecg_dataframe.columns.values.tolist()[label_start_index:]
            assert len(self.diagnoses_cols) > 0, "No labels found in the dataset"
            self.weights = self.compute_weights()
            
        
    def compute_weights(self):
        logger.info("Computing weights...")        
        if len(self.diagnoses_cols) > 1:
            weights = []
            for label in self.diagnoses_cols:
                count = self.ecg_dataframe[label].sum()
                weight = (self.ecg_dataframe.__len__() - count) / (count + 1e-9)
                weights.append(weight)
        else:
            num_labels = self.ecg_dataframe[self.diagnoses_cols[0]].max() + 1
            weights = num_labels / self.ecg_dataframe[self.diagnoses_cols].value_counts()
            weights = weights.values.tolist()
        logger.info("Done with the weights.")
        return torch.FloatTensor(weights)

    
    def __len__(self):
        return len(self.ecg_dataframe)


    def __getitem__(self, idx):
        
        record = self.ecg_dataframe.iloc[idx]
        ecg_filename = record['filename']

        ecg_path = ecg_filename if os.path.isfile(ecg_filename) else os.path.join(self.ecg_dir_path, ecg_filename)

        ecg_data = np.load(ecg_path) # (12, any duration)

        if self.pretrain or self.encode:
            ecg_data = ecg_data[:, :SAMPLES_IN_5_SECONDS_AT_500HZ]
        elif self.random_crop: 
            start = np.random.randint(0, ecg_data.shape[1] - SAMPLES_IN_5_SECONDS_AT_500HZ + 1)
            ecg_data = ecg_data[:, start:start+SAMPLES_IN_5_SECONDS_AT_500HZ]
        elif self.return_full_length:
            # returns a random 10-sec crop since 10 sec is the most common length found in literature and is sufficiently long
            # NOTE: we can't load the entire length (up to 30mins) of an ECG because ECGs from different datasets may have different durations and therefore be unstackable/unbatchable
            # NOTE: HuBERT-ECG is not designed to handle 10-sec ECGs but 5-sec recordings -> the returned ECG must be cropped to 5-sec once returned
            # We use this strategy for TTA
            start = np.random.randint(0, ecg_data.shape[1] - SAMPLES_IN_10_SECONDS_AT_500HZ + 1)
            ecg_data = ecg_data[:, start:start+SAMPLES_IN_10_SECONDS_AT_500HZ]
        else:
            ecg_data = ecg_data[:, :SAMPLES_IN_5_SECONDS_AT_500HZ]
        
        mask = np.isnan(ecg_data)
        ecg_data = np.where(mask, ecg_data[~mask].mean(), ecg_data)
        
        # flatten the leads 
        ecg_data = ecg_data.reshape(-1) # (12*SAMPLES_IN_5_SECONDS_AT_500HZ,)
        
        # downsampling 
        if self.downsampling_factor is not None:
            ecg_data = signal.decimate(ecg_data, self.downsampling_factor)
            
        # compute attention mask
        if not self.encode:
            if self.beat_based_attention_mask:
                attention_mask = self.compute_beat_based_attention_mask(ecg_data)
            else:
                attention_mask = self.compute_attention_mask_for_padding(ecg_data)
        
            
        if self.pretrain:
            
            feat_path = os.path.join(self.features_path, ecg_filename)
            features = np.load(feat_path, allow_pickle=True)                
               
            # [ensamble_length, n_tokens], where values on row i-th are in [0, V_i - 1] and V_i is the number of clusters for the i-th kmeans model
            labels = [kmeans.predict(features).tolist() for kmeans in self.ensamble_kmeans] 
            
            output = (
                torch.from_numpy(ecg_data.copy()).float(),
                torch.from_numpy(attention_mask.copy()).long(),
                torch.Tensor(labels).long()    
            )

            return output
        
        elif self.encode:
            
            return torch.from_numpy(ecg_data.copy()).float(), ecg_filename
        
        else: # finetuning
            labels = record[self.diagnoses_cols].values.astype(float if len(self.diagnoses_cols) > 1 else int)
            output = (
                torch.from_numpy(ecg_data.copy()).float(),
                torch.from_numpy(attention_mask.copy()).long(),
                torch.from_numpy(labels.copy()).float() if len(self.diagnoses_cols) > 1 else torch.from_numpy(labels.copy()).long()
            )
            
            return output
   
    def collate(self, batch : Tuple[Any]):
        unpacked = tuple(zip(*batch))
        if self.encode and not self.pretrain:
            ecg_data = torch.stack(unpacked[0], dim=0)
            ecg_filenames = unpacked[1]
            return ecg_data, ecg_filenames
        else:
            return tuple(map(torch.stack, unpacked))
        
    def compute_attention_mask_for_padding(self, array):
        array = array.reshape(12, -1)     # 12 x SAMPLES_IN_5_SECONDS_AT_500HZ   
        for index in range(array.shape[1]):
            if np.any(array[:, index]):
                break
        start = index
        for index in range(array.shape[1]-1, -1, -1):
            if np.any(array[:, index]):
                break
        end = index
        attention_mask = np.zeros(array.shape[1])
        attention_mask[start:end+1] = 1
        attention_mask = np.repeat([attention_mask], 12, axis=0)
        attention_mask = np.concatenate(attention_mask, axis=0)
        return attention_mask
    
    def compute_beat_based_attention_mask(self, ecg_data):
        ''' 
        Computes attention mask focusing only on P wave, QRS complex and T wave
        '''
        
        ecg_data = ecg_data.reshape(12, SAMPLES_IN_5_SECONDS_AT_500HZ)
        _, rpeaks = nk.ecg_peaks(ecg_data[1], sampling_rate=500) #compute R peaks from II
        signal_dwt, waves_dwt = nk.ecg_delineate(ecg_data[1], rpeaks, sampling_rate=500, method="dwt", show=False, show_type='all')
        signal_dwt['ECG_R_Peaks'] = 0
        signal_dwt['ECG_R_Peaks'].iloc[rpeaks['ECG_R_Peaks']] = 1
        
        p_wave = signal_dwt['ECG_P_Onsets'] | signal_dwt['ECG_P_Offsets'] # binary serie with 1 where P waves start and stop
        qrs_complex = signal_dwt['ECG_Q_Peaks'] | signal_dwt['ECG_S_Peaks'] # binary serie with 1 where QRS complexes start and stop
        t_wave = signal_dwt['ECG_T_Onsets'] | signal_dwt['ECG_T_Offsets'] # binary serie with 1s where T waves start and stop
        
        p_starts_stops = p_wave[p_wave != 0].index.tolist()
        if len(p_starts_stops) % 2 != 0:
            p_starts_stops.append(min(p_starts_stops[-1]+1, 2499))
        p_starts_stops = np.array(p_starts_stops).reshape(-1, 2) # list of couples <start, stop> for each P wave detected
        
        t_starts_stops = t_wave[t_wave != 0].index.tolist()
        if len(t_starts_stops) % 2 != 0:
            t_starts_stops.append(min(t_starts_stops[-1]+1, 2499))
        t_starts_stops = np.array(t_starts_stops).reshape(-1, 2) # list of couples <start, stop> for each T wave detected
        
        
        qrs_starts_stops = qrs_complex[qrs_complex != 0].index.tolist()
        if len(qrs_starts_stops) % 2 != 0:
            qrs_starts_stops.append(min(qrs_starts_stops[-1]+1, 2499))
        qrs_starts_stops = np.array(qrs_starts_stops).reshape(-1, 2) # list of couples <start, stop> for each QRS complex detected
        
        # building the attention mask in order to attend only samples in the p waves
        for start, stop in p_starts_stops:
            p_wave.iloc[start : stop] = 1
        
        # building the attention mask in order to attend only samples in the t waves    
        for start, stop in t_starts_stops:
            t_wave.iloc[start : stop] = 1
        
        # building the attention mask in order to attend only samples in the qrs complexes    
        for start, stop in qrs_starts_stops:
            qrs_complex.iloc[start : stop] = 1
        
        # global attention mask merging all interest regions    
        attention_mask = (p_wave | t_wave | qrs_complex).tolist() 
        attention_mask = np.repeat([attention_mask], 12, axis=0) # since the leads are temporally aligned, interest regions should be located within the same intervals
        attention_mask = np.concatenate(attention_mask, axis=0) 
        
        return attention_mask 
