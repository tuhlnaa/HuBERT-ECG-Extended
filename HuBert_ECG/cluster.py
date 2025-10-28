import joblib
import logging
import os
import wandb

import numpy as np

from pathlib import Path
from rich.logging import RichHandler
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

# Import custom modules
from dataset import ECGDataset
from config import create_clustering_parser, init_seeds

# Constants
NUM_ECG_TOKENS = 93  # Number of ECG embeddings/tokens before the Transformer
RANDOM_SEED = 42

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", 
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


def create_kmeans_model(
    n_clusters: int,
    batch_size: int,
    model_path: Optional[str] = None
) -> MiniBatchKMeans:
    """Create or load a MiniBatchKMeans model.
    
    Args:
        n_clusters: Number of clusters
        batch_size: Batch size for mini-batch k-means
        model_path: Path to pre-trained model to resume from
        
    Returns:
        MiniBatchKMeans model instance
        
    Raises:
        AssertionError: If loaded model has different number of clusters
    """
    if model_path is not None:
        logger.info(f"Loading pre-trained model from {model_path}")
        model = joblib.load(model_path)
        n_loaded_clusters = model.cluster_centers_.shape[0]
        assert n_clusters == n_loaded_clusters, (
            f"Resume clustering failed. Loaded model has {n_loaded_clusters} clusters, "
            f"expected {n_clusters}"
        )
        return model
    
    logger.info("Creating clustering model from scratch")
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_SEED,
        compute_labels=True,
        batch_size=batch_size * NUM_ECG_TOKENS,
        n_init=20,
        max_no_improvement=100,
        reassignment_ratio=0.0
    )


def generate_model_filename(
    n_clusters: int,
    train_iteration: int,
    layer: Optional[int],
    sse: float
) -> str:
    """Generate a descriptive filename for the clustering model.
    
    Args:
        n_clusters: Number of clusters
        train_iteration: Training iteration number
        layer: Encoder layer (None for morphology-based clustering)
        sse: Sum of squared errors
        
    Returns:
        Model filename
    """
    if train_iteration == 1:
        base_name = f"k_means_{n_clusters}_morphology"
    else:
        base_name = f"k_means_{n_clusters}_encoder_{layer}_{train_iteration}"
    
    sse_str = f"{int(sse):e}"
    return f"{base_name}_{sse_str}.pkl"


def cluster(args) -> None:
    """Perform k-means clustering on ECG features.
    
    Args:
        args: Arguments from argument parser containing clustering configuration
    """
    group = f"clustering_iteration_{args.train_iteration}"
    wandb.init(project="HuBert ECG", group=group, entity=None)
    
    dataset = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv,
        ecg_dir_path=args.in_dir,
        pretrain=False,
        encode=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    
    normalizer = preprocessing.Normalizer()
    n_clusters = args.n_clusters_start
    global_step = 0
    
    # Resume from checkpoint only on first iteration
    model_path = args.model_path if global_step == 0 else None
    
    while n_clusters <= args.n_clusters_end:
        global_step += 1
        logger.info(f"Running k-means with {n_clusters} clusters...")
        
        model = create_kmeans_model(n_clusters, args.batch_size, model_path)
        model_path = None  # Only load once at start
        
        # Fitting loop
        for _, filenames in tqdm(dataloader, total=len(dataloader), desc=f"k={n_clusters}"):
            features = load_and_normalize_features(filenames, args.in_dir, normalizer)
            model.partial_fit(features)
        
        # Log and save results
        sse = model.inertia_
        wandb.log({"k": n_clusters, "SSE": sse}, step=global_step)
        
        model_filename = generate_model_filename(
            n_clusters, args.train_iteration, args.layer, sse
        )
        joblib.dump(model, model_filename)
        
        n_clusters += args.step


def load_and_normalize_features(
    filenames: list[str], 
    feature_dir: str, 
    normalizer: preprocessing.Normalizer
) -> np.ndarray:
    """Load features from files and normalize them.
    
    Args:
        filenames: List of feature filenames to load
        feature_dir: Directory containing feature files
        normalizer: Fitted normalizer for feature normalization
        
    Returns:
        Normalized feature array, shape=(Batch size * 93, Number of features)
    """
    features = [np.load(os.path.join(feature_dir, filename)) for filename in filenames]
    features = np.concatenate(features, axis=0)
    return normalizer.transform(features)


def evaluate_clustering(args) -> None:
    """Evaluate a trained clustering model using Davies-Bouldin and Calinski-Harabasz scores.
    
    Args:
        args: Arguments from argument parser containing evaluation configuration
    """
    model_name = Path(args.model_path).name
    logger.info(f"Evaluating clustering model: {model_name}")
    
    dataset = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv,
        ecg_dir_path=args.in_dir,
        pretrain=False,
        encode=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    
    model = joblib.load(args.model_path)
    normalizer = preprocessing.Normalizer()
    
    db_scores = []
    ch_scores = []
    
    for _, filenames in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
        features = load_and_normalize_features(filenames, args.in_dir, normalizer)
        print(features.shape)
        assignments = model.predict(features)
        
        db_scores.append(davies_bouldin_score(features, assignments))
        ch_scores.append(calinski_harabasz_score(features, assignments))
    
    logger.info(f"Average Davies-Bouldin score: {np.mean(db_scores):.4f}")
    logger.info(f"Average Calinski-Harabasz score: {np.mean(ch_scores):.4f}")

    # Average Davies-Bouldin score: 2.0163889348431745
    # Average Calinski-Harabasz score: 62.601320774361625


def main() -> None:
    """Main entry point for clustering script."""
    args = create_clustering_parser()
    init_seeds(seed=RANDOM_SEED)
    
    if args.cluster:
        cluster(args)
    else:
        evaluate_clustering(args)


if __name__ == "__main__":
    main()