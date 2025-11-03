"""
K-Means Clustering Module for ECG Feature Learning

This module provides functionality for performing k-means clustering on ECG 
features using a HuBERT-style approach. It supports iterative clustering with 
configurable cluster ranges, model checkpointing, and evaluation metrics 
(Davies-Bouldin and Calinski-Harabasz scores).

Usage:
# Train clustering model
python ./HuBert_ECG/kmeans_clustering.py /path/to/dataset.csv /path/to/features 1 32 --cluster --n_clusters_start 100 --n_clusters_end 500 --step 100

# Resume training from checkpoint
python ./HuBert_ECG/kmeans_clustering.py /path/to/dataset.csv /path/to/features 2 32 --cluster --n_clusters_start 100 --n_clusters_end 500 --step 100 --model_path kmeans_100_morphology_sse1e+06.pkl --layer 6

# Evaluate trained model
python ./HuBert_ECG/kmeans_clustering.py /path/to/dataset.csv /path/to/features 1 32 --model_path kmeans_500_morphology_sse8e+05.pkl
"""
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
from typing import List, Optional, Dict

# Import custom modules
from config import create_clustering_parser, init_seeds
from dataset import ECGDataset
from logging_utils import ClearMLLogger, SimplePhasedMetricHandler

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


def compute_clustering_metrics(
    model: MiniBatchKMeans,
    dataloader: DataLoader,
    feature_dir: Path,
    normalizer: Normalizer,
    desc: str = "Computing metrics"
) -> Dict[str, float]:
    """Compute clustering evaluation metrics.
    
    Args:
        model: Trained k-means model
        dataloader: DataLoader for the dataset
        feature_dir: Directory containing feature files
        normalizer: Normalizer for features
        desc: Description for progress bar
        
    Returns:
        Dictionary containing average DB score, CH score, and SSE
    """
    db_scores = []
    ch_scores = []
    batch_sses = []
    
    for _, filenames in tqdm(dataloader, total=len(dataloader), desc=desc):
        features = load_and_normalize_features(filenames, feature_dir, normalizer)
        assignments = model.predict(features)
        
        # Compute metrics
        db_scores.append(davies_bouldin_score(features, assignments))
        ch_scores.append(calinski_harabasz_score(features, assignments))
        
        # Compute batch SSE (not use model.inertia_)
        centers = model.cluster_centers_[assignments]
        batch_sse = np.sum((features - centers) ** 2)
        batch_sses.append(batch_sse)
    
    return {
        "db_score": np.mean(db_scores),
        "ch_score": np.mean(ch_scores),
        "sse": np.sum(batch_sses)
    }


def cluster(args) -> None:
    """Perform k-means clustering on ECG features with train and validation evaluation.
    
    Args:
        args: Arguments from argument parser containing clustering configuration
    """
    save_dir = Path(f"{args.output_dir}/sklearn-model")
    save_dir.mkdir(parents=True, exist_ok=True)

    group = f"clustering_iteration_{args.train_iteration}"
    wandb.init(project="HuBert ECG", group=group, entity=None)

    clearml_config = {
        'project': "HuBERT-ECG",  # ClearML project name
        'task_name': "HuBERT-ECG clustering",  
        'task_type': "training",  # ClearML task type
        'reuse_last_task_id': False,  # ClearML task ID to resume or boolean flag
        "tags": ["clustering"],  # List of tags for ClearML task
    }

    # ClearML uses 1337 as the default initial seed
    clearml_logger = ClearMLLogger(args.output_dir, **clearml_config)
    clearml_logger.register_handler(SimplePhasedMetricHandler(['sse', 'db_score', 'ch_score']))

    # Upload args as JSON
    clearml_logger.log_args_as_json(args)
    
    init_seeds(seed=RANDOM_SEED)

    # Create train dataset
    train_dataset = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv_train,
        ecg_dir_path=args.in_dir,
        pretrain=False,
        encode=True
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    
    # Create validation dataset
    val_dataset = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv_val,
        ecg_dir_path=args.in_dir,
        pretrain=False,
        encode=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
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
        
        # Training loop
        for _, filenames in tqdm(train_dataloader, total=len(train_dataloader), desc=f"Training k={n_clusters}"):
            features = load_and_normalize_features(filenames, feature_dir, normalizer)
            model.partial_fit(features)
        
        # Evaluate on training set
        logger.info("Evaluating on training set...")
        train_metrics = compute_clustering_metrics(
            model, train_dataloader, feature_dir, normalizer, 
            desc="Train evaluation"
        )
        
        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        val_metrics = compute_clustering_metrics(
            model, val_dataloader, feature_dir, normalizer,
            desc="Val evaluation"
        )
        
        # Log metrics to W&B
        wandb.log({
            "train_sse": train_metrics["sse"],
            "train_db_score": train_metrics["db_score"],
            "train_ch_score": train_metrics["ch_score"],
            "val_sse": val_metrics["sse"],
            "val_db_score": val_metrics["db_score"],
            "val_ch_score": val_metrics["ch_score"]
        }, step=n_clusters)
        
        clearml_logger.log_metrics({
            "train_sse": train_metrics["sse"],
            "train_db_score": train_metrics["db_score"],
            "train_ch_score": train_metrics["ch_score"],
            "val_sse": val_metrics["sse"],
            "val_db_score": val_metrics["db_score"],
            "val_ch_score": val_metrics["ch_score"]
        }, n_clusters)

        # Log results
        logger.info(f"Train - SSE: {train_metrics['sse']:.2f}, DB: {train_metrics['db_score']:.4f}, CH: {train_metrics['ch_score']:.2f}")
        logger.info(f"Val   - SSE: {val_metrics['sse']:.2f}, DB: {val_metrics['db_score']:.4f}, CH: {val_metrics['ch_score']:.2f}")
        
        # Save model with validation SSE in filename
        model_filename = generate_model_filename(
            n_clusters, args.train_iteration, args.layer, val_metrics["sse"]
        )
        joblib.dump(model, save_dir / model_filename)
        logger.info(f"Saved model to {model_filename}")
        
        n_clusters += args.step


def evaluate_clustering(args) -> None:
    """Evaluate a trained clustering model using Davies-Bouldin, Calinski-Harabasz scores, and SSE.
    
    Args:
        args: Arguments from argument parser containing evaluation configuration
    """
    init_seeds(seed=RANDOM_SEED)
    model_path = Path(args.model_path)
    logger.info(f"Evaluating clustering model: {model_path.name}")
    
    dataset = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv_val,
        ecg_dir_path=args.in_dir,
        pretrain=False,
        encode=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )
    
    model = joblib.load(model_path)
    normalizer = Normalizer()
    feature_dir = Path(args.in_dir)
    
    # Compute all metrics
    metrics = compute_clustering_metrics(
        model, dataloader, feature_dir, normalizer,
        desc="Evaluating"
    )
    
    logger.info(f"Sum of Squared Errors (SSE): {metrics['sse']:.2f}")
    logger.info(f"Average Davies-Bouldin score: {metrics['db_score']:.4f} (lower is better)")
    logger.info(f"Average Calinski-Harabasz score: {metrics['ch_score']:.4f} (higher is better)")


def main() -> None:
    """Main entry point for clustering script."""
    args = create_clustering_parser()

    if args.cluster:
        cluster(args)
    else:
        evaluate_clustering(args)


if __name__ == "__main__":
    main()