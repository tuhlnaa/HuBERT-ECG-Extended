"""K-means clustering for ECG feature learning using HuBERT-style approach."""

import joblib
import logging
import wandb
import numpy as np

from pathlib import Path
from rich.logging import RichHandler
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import Normalizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from typing import List, Optional

# Import custom modules
from config import create_clustering_parser, init_seeds
from dataset import ECGDataset

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
    model_path: Optional[Path] = None
) -> MiniBatchKMeans:
    """Create or load a MiniBatchKMeans model.
    
    Args:
        n_clusters: Number of clusters
        batch_size: Batch size for mini-batch k-means
        model_path: Path to pre-trained model to resume from
        
    Returns:
        MiniBatchKMeans model instance
    """
    if model_path is not None:
        logger.info(f"Loading pre-trained model from {model_path}")
        model = joblib.load(model_path)
        n_loaded_clusters = model.cluster_centers_.shape[0]
        if n_clusters != n_loaded_clusters:
            raise ValueError(
                f"Resume clustering failed. Loaded model has {n_loaded_clusters} "
                f"clusters, expected {n_clusters}"
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
        base_name = f"kmeans_{n_clusters}_morphology"
    else:
        base_name = f"kmeans_{n_clusters}_encoder_l{layer}_iter{train_iteration}"
    
    sse_str = f"{int(sse):e}"
    return f"{base_name}_sse{sse_str}.pkl"


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
    
    normalizer = Normalizer()
    feature_dir = Path(args.in_dir)
    n_clusters = args.n_clusters_start
    global_step = 0
    
    # Resume from checkpoint if provided
    initial_model_path = Path(args.model_path) if args.model_path else None
    
    while n_clusters <= args.n_clusters_end:
        global_step += 1
        logger.info(f"Running k-means with {n_clusters} clusters...")
        
        # Only load checkpoint for first clustering run
        model_path = initial_model_path if global_step == 1 else None
        model = create_kmeans_model(n_clusters, args.batch_size, model_path)
        
        # Fitting loop
        for _, filenames in tqdm(dataloader, total=len(dataloader), desc=f"k={n_clusters}"):
            features = load_and_normalize_features(filenames, feature_dir, normalizer)
            model.partial_fit(features)
        
        # Log and save results
        sse = model.inertia_
        wandb.log({"k": n_clusters, "SSE": sse}, step=global_step)
        
        model_filename = generate_model_filename(
            n_clusters, args.train_iteration, args.layer, sse
        )
        joblib.dump(model, model_filename)
        logger.info(f"Saved model to {model_filename}")
        
        n_clusters += args.step


def load_and_normalize_features(
    filenames: List[str],
    feature_dir: Path,
    normalizer: Normalizer
) -> np.ndarray:
    """Load features from files and normalize them.
    
    Args:
        filenames: List of feature filenames to load
        feature_dir: Directory containing feature files
        normalizer: Normalizer for feature normalization (L2 norm)
        
    Returns:
        Normalized feature array with shape (batch_size * NUM_ECG_TOKENS, n_features)
    """
    features = [np.load(feature_dir / filename) for filename in filenames]
    features = np.concatenate(features, axis=0)
    return normalizer.transform(features)


def evaluate_clustering(args) -> None:
    """Evaluate a trained clustering model using Davies-Bouldin and Calinski-Harabasz scores.
    
    Args:
        args: Arguments from argument parser containing evaluation configuration
    """
    model_path = Path(args.model_path)
    logger.info(f"Evaluating clustering model: {model_path.name}")
    
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
        shuffle=False,  # No need to shuffle for evaluation
        pin_memory=True,
        drop_last=True
    )
    
    model = joblib.load(model_path)
    normalizer = Normalizer()
    feature_dir = Path(args.in_dir)
    
    db_scores = []
    ch_scores = []
    
    for _, filenames in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
        features = load_and_normalize_features(filenames, feature_dir, normalizer)
        assignments = model.predict(features)
        
        db_scores.append(davies_bouldin_score(features, assignments))
        ch_scores.append(calinski_harabasz_score(features, assignments))
    
    logger.info(f"Average Davies-Bouldin score: {np.mean(db_scores):.4f} (lower is better)")
    logger.info(f"Average Calinski-Harabasz score: {np.mean(ch_scores):.4f} (higher is better)")


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

"""
Average Davies-Bouldin score: 2.0157
Average Calinski-Harabasz score: 62.3056
"""
