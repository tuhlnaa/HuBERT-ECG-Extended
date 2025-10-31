import copy
import logging
import torch
import wandb
import numpy as np
import torch.optim as optim

from loguru import logger
from pathlib import Path
from rich.logging import RichHandler
from tqdm import tqdm
from torch.nn import functional as F

# Import custom modules
from config import create_training_parser, init_seeds
from dataset import create_dataloader
from validator import validate_pretrain_model
from training_utils import (
    _create_lr_scheduler, 
    _create_optimizer, 
    _ensure_min_dropout, 
    _get_model_config, 
    initialize_model_from_scratch, 
    resume_from_checkpoint, 
    _validate_vocab_sizes, 
    dynamic_regularizer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", 
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

EPS = 1E-09
MINIMAL_IMPROVEMENT = 1e-3
DROPOUT_ADJUSTMENT = 0.05
WEIGHT_DECAY_MULTIPLIER = 5.0
SELF_SUPERVISED_MODEL_CKPT_PATH = "output/checkpoints/self-supervised/"
        
# Constants
WARMUP_RATIO = 0.08
DROPOUT_RESET_VALUE = 0.1
MIN_WEIGHT_DECAY = 0.01


def _validate_vocab_sizes(args, dataset):
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


def train(args):
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize tracking
    wandb.init(
        project="HuBert ECG",
        group="self-supervised",
        entity=None,
        name=args.wandb_run_name
    )

    # Get model configuration
    model_config = _get_model_config(args.largeness)
    
    ### configs ###
    patience = args.patience if args.patience is not None else args.training_steps // args.val_interval
    lr = args.lr
    betas = (0.9, 0.98)
    weight_decay = max(0, 0.01 * args.weight_decay_mult)
    accumulation_steps = args.accumulation_steps
    mask_time_prob = args.mask_time_prob
    
    scaler = torch.amp.GradScaler()
    
    # Setup data
    train_loader = create_dataloader(
        csv_path=args.path_to_dataset_csv_train,
        ecg_dir=args.ecg_dir_path,
        batch_size=args.batch_size,
        downsample_factor=args.downsampling_factor,
        features_path=args.train_features_path,
        kmeans_path=args.kmeans_path,
        is_pretrain=True,
        drop_last=False
    )
    val_loader = create_dataloader(
        csv_path=args.path_to_dataset_csv_val,
        ecg_dir=args.ecg_dir_path,
        batch_size=args.batch_size,
        downsample_factor=args.downsampling_factor,
        features_path=args.val_features_path,
        kmeans_path=args.kmeans_path,
        is_pretrain=True,
        shuffle=False,
        drop_last=False
    )

    # Validate configuration
    _validate_vocab_sizes(args, train_loader.dataset)

    if args.training_steps is not None:
        steps_per_epoch = len(train_loader) // accumulation_steps
        epochs =  args.training_steps // steps_per_epoch + 1
    else:
        epochs = args.epochs



    if args.resume_pretraining:
        model, training_state = resume_from_checkpoint(args, device)
    else:
        model, training_state = initialize_model_from_scratch(args, model_config, mask_time_prob, device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=EPS, weight_decay=weight_decay)

    if args.resume_pretraining:
        optimizer.load_state_dict(training_state['optimizer_state'])
        optimizer.param_groups[0]['weight_decay'] = max(MIN_WEIGHT_DECAY, 
                                                         optimizer.param_groups[0]['weight_decay'])
        _ensure_min_dropout(model, DROPOUT_RESET_VALUE)
    
    lr_scheduler = _create_lr_scheduler(
        optimizer, 
        args.training_steps, 
        WARMUP_RATIO,
        training_state['global_step'],
        training_state.get('scheduler_state')
    )

    hubert = model
    global_step = training_state['global_step']
    best_val_loss = training_state['best_val_loss']
    best_val_accuracy = training_state['best_val_accuracy']
    patience_count = training_state['patience_count']

    start_epoch = global_step // len(train_loader)

    for epoch in range(start_epoch, epochs):
        hubert.train()
        logger.info(f"Epoch {epoch+1}/{epochs}")

        train_losses = []
        
        # Store the tqdm iterator in a variable
        pbar = tqdm(train_loader, total=len(train_loader))

        for ecg, attention_mask, ensemble_labels in pbar:
            global_step += 1
            
            # Move data to device
            ecg = ecg.to(device) 
            attention_mask = attention_mask.to(device)
            ensemble_labels = ensemble_labels.to(device)

            hubert.train()

            with torch.amp.autocast('cuda'):
                # Forward pass
                encoder_output = hubert(
                    ecg, 
                    attention_mask=attention_mask, 
                    output_attentions=False, 
                    output_hidden_states=False, 
                    return_dict=True
                )
                
                mask = encoder_output['mask_time_indices']
                ensemble_logits = hubert.logits(encoder_output['last_hidden_state'])

                # Compute ensemble loss
                ensemble_labels = ensemble_labels.transpose(0, 1)
                
                # Vectorized loss computation
                masked_losses = []
                unmasked_losses = []

                for labels, logits in zip(ensemble_labels, ensemble_logits):
                    # labels: (batch_size, seq_len), logits: (batch_size, seq_len, vocab_size)
                    masked_losses.append(F.cross_entropy(logits[mask], labels[mask]))
                    unmasked_losses.append(F.cross_entropy(logits[~mask], labels[~mask]))
                
                masked_loss = sum(masked_losses)
                unmasked_loss = sum(unmasked_losses)
                
                loss = args.alpha * masked_loss + (1 - args.alpha) * unmasked_loss
                loss = loss / accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            train_losses.append(loss.item())

            # Update progress bar with global_step and loss
            pbar.set_postfix({'global_step': global_step})

            # Gradient accumulation
            if global_step % accumulation_steps == 0:
                scaler.step(optimizer)
                lr_scheduler.step()
                scaler.update()
                optimizer.zero_grad()


            # Validation every val_interval steps
            if global_step % args.val_interval == 0:
                val_loss, val_accuracy = validate_pretrain_model(
                    hubert, val_loader, device, logger, global_step
                )
                
                train_loss = np.mean(train_losses)
                train_losses.clear()

                # Logging
                logger.info(f"Step: {global_step}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.4f}")
                                                                        
                wandb.log({
                    f"train_loss": train_loss,
                    f"val_loss": val_loss,
                    "val_accuracy": val_accuracy
                }, step=global_step)

                checkpoint_path = Path(SELF_SUPERVISED_MODEL_CKPT_PATH)
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                            
                # Save if new best model + Early stopping
                loss_improved = val_loss <= best_val_loss - MINIMAL_IMPROVEMENT
                accuracy_improved = val_accuracy >= best_val_accuracy + MINIMAL_IMPROVEMENT
                
                if loss_improved or accuracy_improved:
                    # Update best metrics
                    if loss_improved:
                        best_val_loss = val_loss
                        best_val_accuracy = max(val_accuracy, best_val_accuracy)
                        patience_count = 0
                        logger.info(
                            f"New best (best_val_loss={best_val_loss:.4f}) - "
                            f"model saved at step {global_step}"
                        )
                    else:  # accuracy_improved only
                        best_val_accuracy = val_accuracy
                        logger.info(
                            f"Val loss not improved but val accuracy did "
                            f"(best_val_accuracy={best_val_accuracy:.4f}) - "
                            f"model saved at step {global_step}"
                        )
                    
                    # Create checkpoint
                    checkpoint = {
                        "global_step": global_step,
                        "patience_count": patience_count,
                        "model_config": hubert.config,
                        "model_state_dict": copy.deepcopy(hubert.state_dict()),
                        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                        "scheduler_state_dict": copy.deepcopy(lr_scheduler.state_dict()),
                        "best_val_loss": best_val_loss,
                        "best_val_accuracy": best_val_accuracy,
                        "pretraining_vocab_sizes": args.vocab_sizes,
                    }
                    
                    checkpoint_name = (
                        f"hubert_{args.train_iteration}_iteration_"
                        f"{global_step}.pt"
                    )
                    # torch.save(checkpoint, checkpoint_path / checkpoint_name)

                    if global_step == 61:
                        torch.save(checkpoint, checkpoint_path / checkpoint_name)
                    
                    # Reduce regularization after improvement
                    if args.dynamic_reg:
                        dynamic_regularizer(optimizer, hubert, penalty=False)
                
                else:  # No improvement
                    patience_count += 1
                    
                    # Apply regularization penalty at intervals
                    if args.dynamic_reg and patience_count != patience:
                        if patience_count % (patience // args.intervals_for_penalty) == 0:
                            dynamic_regularizer(optimizer, hubert, penalty=True)
                    
                    # Early stopping
                    if patience_count >= patience:
                        logger.warning(
                            f"EARLY STOPPING: Max patience reached at step {global_step} "
                            f"(patience_count={patience_count})"
                        )
                        return

    # End of training iteration
    logger.info("End of training")
    logger.info(f"STATS: Global step={global_step}, Best val loss={best_val_loss:.4f}")
    wandb.finish()


def main() -> None:
    args = create_training_parser()
    init_seeds(seed=42)
    train(args)


if __name__ == "__main__":
    main()
