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
import matplotlib.pyplot as plt

from pathlib import Path
from rich.logging import RichHandler
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import Normalizer
from sklearn.manifold import TSNE

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


def visualize_tsne(
    model: MiniBatchKMeans,
    dataloader: DataLoader,
    feature_dir: Path,
    normalizer: Normalizer,
    save_path: Path,
    n_samples: int = 10000,
    perplexity: int = 30,
    desc: str = "Computing T-SNE"
) -> None:
    """Create T-SNE visualization of clustered features.
    
    Args:
        model: Trained k-means model
        dataloader: DataLoader for the dataset
        feature_dir: Directory containing feature files
        normalizer: Normalizer for features
        save_path: Path to save the visualization
        n_samples: Maximum number of samples to use for T-SNE (for performance)
        perplexity: T-SNE perplexity parameter
        desc: Description for progress bar
    """
    all_features = []
    all_labels = []
    total_samples = 0
    
    # Collect features and cluster assignments
    for _, filenames in tqdm(dataloader, total=len(dataloader), desc=desc):
        if total_samples >= n_samples:
            break
            
        features = load_and_normalize_features(filenames, feature_dir, normalizer)
        assignments = model.predict(features)
        
        # Limit samples to avoid memory issues
        samples_to_take = min(len(features), n_samples - total_samples)
        all_features.append(features[:samples_to_take])
        all_labels.append(assignments[:samples_to_take])
        total_samples += samples_to_take
    
    # Concatenate all collected features
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    logger.info(f"Computing T-SNE for {len(all_features)} samples...")
    
    # Compute T-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    tsne_features = tsne.fit_transform(all_features)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        tsne_features[:, 0],
        tsne_features[:, 1],
        c=all_labels,
        cmap='tab20',
        alpha=0.6,
        s=10
    )
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'T-SNE Visualization of Clusters (n={model.n_clusters})')
    plt.xlabel('T-SNE Component 1')
    plt.ylabel('T-SNE Component 2')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved T-SNE visualization to {save_path}")


def cluster(args) -> None:
    """Perform k-means clustering on ECG features with train and validation evaluation.
    
    Args:
        args: Arguments from argument parser containing clustering configuration
    """
    # Create directory
    save_dir = Path(f"{args.output_dir}/sklearn-model")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    tsne_dir = Path(f"{args.output_dir}/tsne_visualizations")
    tsne_dir.mkdir(parents=True, exist_ok=True)

    # group = f"clustering_iteration_{args.train_iteration}"
    # wandb.init(project="HuBert ECG", group=group, entity=None)

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
        
        # Generate T-SNE visualizations
        logger.info("Generating T-SNE visualizations...")
        
        # Train T-SNE
        train_tsne_path = tsne_dir / f"frame_{global_step:03d}_tsne_train_{n_clusters}_clusters.png"
        visualize_tsne(
            model, train_dataloader, feature_dir, normalizer,
            save_path=train_tsne_path,
            desc="T-SNE (train)"
        )
        
        # # Validation T-SNE
        # val_tsne_path = tsne_dir / f"tsne_val_{n_clusters}_clusters.png"
        # visualize_tsne(
        #     model, val_dataloader, feature_dir, normalizer,
        #     save_path=val_tsne_path,
        #     desc="T-SNE (val)"
        # )
        
        # # Log T-SNE images to ClearML
        # clearml_logger.log_image("train_tsne", str(train_tsne_path), n_clusters)
        # clearml_logger.log_image("val_tsne", str(val_tsne_path), n_clusters)
        
        # # Log metrics to W&B
        # wandb.log({
        #     "train_sse": train_metrics["sse"],
        #     "train_db_score": train_metrics["db_score"],
        #     "train_ch_score": train_metrics["ch_score"],
        #     "val_sse": val_metrics["sse"],
        #     "val_db_score": val_metrics["db_score"],
        #     "val_ch_score": val_metrics["ch_score"]
        # }, step=n_clusters)

        # Log metrics to ClearML
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
        if args.train_iteration == 1:
            model_filename = f"kmeans_{n_clusters}_morphology.pkl"
        else:
            model_filename = f"kmeans_{n_clusters}_encoder_l{args.layer}_iter{args.train_iteration}.pkl"

        joblib.dump(model, save_dir / model_filename)
        logger.info(f"Saved model to {model_filename}")
        
        n_clusters += args.step


def evaluate_clustering(args) -> None:
    """Evaluate a trained clustering model using Davies-Bouldin, Calinski-Harabasz scores, SSE, and T-SNE visualization.
    
    Args:
        args: Arguments from argument parser containing evaluation configuration
    """
    init_seeds(seed=RANDOM_SEED)
    feature_dir = Path(args.in_dir)
    model_path = Path(args.model_path)
    logger.info(f"Evaluating clustering model: {model_path.name}")
    
    # Create directory for T-SNE visualization
    tsne_dir = Path(args.output_dir) / "tsne_visualizations"
    tsne_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Compute all metrics
    metrics = compute_clustering_metrics(
        model, dataloader, feature_dir, normalizer,
        desc="Evaluating"
    )
    
    logger.info(f"Sum of Squared Errors (SSE): {metrics['sse']:.2f}")
    logger.info(f"Average Davies-Bouldin score: {metrics['db_score']:.4f} (lower is better)")
    logger.info(f"Average Calinski-Harabasz score: {metrics['ch_score']:.4f} (higher is better)")
    
    # Generate T-SNE visualization
    logger.info("Generating T-SNE visualization...")
    tsne_path = tsne_dir / f"tsne_evaluation_{model.n_clusters}_clusters.png"
    visualize_tsne(
        model, dataloader, feature_dir, normalizer,
        save_path=tsne_path,
        desc="Computing T-SNE"
    )


def main() -> None:
    """Main entry point for clustering script."""
    args = create_clustering_parser()

    if args.cluster:
        cluster(args)
    else:
        evaluate_clustering(args)


if __name__ == "__main__":
    main()