import joblib
import os
import random
import wandb
import numpy as np

from loguru import logger
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

# os.environ["OPENBLAS_NUM_THREADS"] = "4"

# Import custom modules
from dataset import ECGDataset
from config import create_clustering_parser


def cluster(args):
    
    group = f"clustering_iteration_{args.train_iteration}"
   
    wandb.init(entity="your-entity", project="your-project", group=group)

    np.random.seed(42)
    random.seed(42)
    
    data_set = ECGDataset(
        path_to_dataset_csv = args.path_to_dataset_csv,
        ecg_dir_path = args.in_dir,
        pretrain = False,
        encode = True
    )
  
    dataloader = DataLoader(
        data_set,
        batch_size = args.batch_size,
        num_workers=5,
        shuffle = True,
        pin_memory = True,
        drop_last=True,
        prefetch_factor=4
    )
    
    n_clusters = args.n_clusters_start

    normalizer = preprocessing.Normalizer()
    
    while n_clusters <= args.n_clusters_end:
        
        logger.info(f"Running kmeans with {n_clusters} clusters...")
        already_loaded = False # one can resume one model fitting at a time. After resumed fitting is terminated, if while is not ended, a new clustering model is initialized for clustering

        if args.model_path is not None and not already_loaded:
            logger.info("Loading a partially fitted model...")
            model = joblib.load(args.model_path)
            n_loaded_clusters = model.cluster_centers_.shape[0]
            assert n_clusters == n_loaded_clusters, f"Resume clustering failed. Loaded model clusters ({n_loaded_clusters}) not equal to current n_clusters ({n_clusters})"
            already_loaded = True
        else:
            logger.info("Creating a clustering model from scratch...")
            # model creation with same batch size as in the dataloader
            model = MiniBatchKMeans(
                n_clusters = n_clusters,
                random_state = 42,
                compute_labels = True,
                batch_size = args.batch_size * 93, # 93 is the number of ECG embeddings/tokens before the Transformer
                n_init = 20,
                max_no_improvement = 100,
                reassignment_ratio = 0.0
            )
        
        # retrive features from dataloader
        
        ### FITTING LOOP ###
        
        for _, filenames in tqdm(dataloader, total = len(dataloader)):

            features = [np.load(os.path.join(args.in_dir, filename)) for filename in filenames] 

            # build the batch from filenames returned by the dataloader                
            features = np.concatenate(features, axis = 0) 
            
            # normalize features in the batch
            features = normalizer.transform(features)
            
            # train kmeans
            model.partial_fit(features) 
            
        ### END OF FITTING LOOP ###
            
        sse = model.inertia_
        wandb.log({"k" : n_clusters, "SSE" : sse})
        

        if args.train_iteration == 1:
            model_name = "k_means_" + str(n_clusters) +  "_morphology"
        else:
            model_name = "k_means_" + str(n_clusters) + "_encoder_" + str(args.layer) + "_" + str(args.train_iteration)
            
        sse = "{:e}".format(int(sse))
        
        logger.info(f"SSE: {sse}")
        
        model_name +=  "_" + sse + ".pkl"
        
        joblib.dump(model, os.path.join("./", model_name))
        logger.info(f"{model_name} model saved.\n")
        
        n_clusters = n_clusters + args.step
 
def evaluate_clustering(args):
    
    logger.info(f"Ready to evaluate clustering model {args.model_path.split('/')[-1]}")
    
    group = f"clustering_iteration_{args.train_iteration}_evaluation"
    
    wandb.init(entity="your-project", project="your-project", group=group)
    
    
    
    np.random.seed(42)
    random.seed(42)
    
    data_set = ECGDataset(
        path_to_dataset_csv = args.path_to_dataset_csv,
        ecg_dir_path = args.in_dir,
        pretrain = False,
        encode = True
    )
    
  
    dataloader = DataLoader(
        data_set,
        batch_size = args.batch_size,
        num_workers=5,
        shuffle = True,
        pin_memory = True,
        drop_last=True
    )
    
    model = joblib.load(args.model_path)
    
    db_scores = []
    ch_scores = []
    
    for _, filenames in tqdm(dataloader, total = len(dataloader)):

            features = [np.load(os.path.join(args.in_dir, filename)) for filename in filenames] 
            
            # build the batch from filenames returned by the dataloader                
            features = np.concatenate(features, axis = 0)
            
            # normalize features in the batch
            features = preprocessing.Normalizer().fit_transform(features)
            
            assignments = model.predict(features) # (BS * 93, ) containing values in [0, n_clusters - 1]
            
            db_scores.append(davies_bouldin_score(features, assignments))
            ch_scores.append(calinski_harabasz_score(features, assignments))
            
    logger.info(f"Average Davies-Bouldin score: {np.mean(db_scores)}")
    logger.info(f"Average Calinski-Harabasz score: {np.mean(ch_scores)}")
    wandb.log({"Average Davies-Bouldin score" : np.mean(db_scores)})
    wandb.log({"Average Calinski-Harabsz score" : np.mean(ch_scores)})


def main():
    args = create_clustering_parser()

    if args.cluster:
        cluster(args)
    else:
        evaluate_clustering(args)


if __name__ == "__main__":
    main()
