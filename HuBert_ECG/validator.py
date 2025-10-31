import torch
import numpy as np

from tqdm import tqdm
from typing import Dict, Any
from torch.nn import functional as F

class Validator:
    """Validation engine for model evaluation."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        metrics: Any,
        device: torch.device,
        target_metric: str
    ):
        self.model = model
        self.val_loader = val_loader
        self.criterion = criterion
        self.metrics = metrics
        self.device = device
        self.target_metric = target_metric


    def validate(self) -> Dict[str, float]:
        """Run validation loop and return metrics."""
        self.model.eval()
        self.metrics.reset()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, total=len(self.val_loader), desc="Validation"):
                ecg, _, labels = batch
                ecg = ecg.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                # Forward pass
                logits, _ = self.model(
                    ecg, 
                    attention_mask=None, 
                    output_attentions=False, 
                    output_hidden_states=False, 
                    return_dict=False
                )
                # Compute loss
                loss = self.criterion(logits, labels)
                
                self.metrics.update(logits, labels, loss)
        
        # Compute all metrics
        metrics_dict = self.metrics.compute()
        target_score = self.metrics.get_target_metric(self.target_metric)
        
        return {
            **metrics_dict,
            'target_score': target_score
        }


def validate_pretrain_model(model, val_loader, device, logger, global_step):
    """Validation loop extracted as separate function."""
    model.eval()
    
    val_losses = []
    val_accuracies = []
    
    logger.info(f"Validating model at step {global_step}...")
    
    with torch.no_grad():
        for ecg, _, ensemble_labels in tqdm(val_loader, total=len(val_loader)):
            ecg, ensemble_labels = ecg.to(device), ensemble_labels.to(device)
            # attention_mask = (attention_mask).to(device) # attention mask could harm inference performance according to HF docs

            # (ensamble_length, batch_size, sequence_length)
            ensemble_labels = ensemble_labels.transpose(0, 1)

            # Forward pass (no attention mask during validation per HF recommendation)
            encoder_output = model(
                ecg, 
                attention_mask=None, 
                output_attentions=False, 
                output_hidden_states=False, 
                return_dict=True
            )
            ensemble_logits = model.logits(encoder_output['last_hidden_state'])

            assert len(ensemble_labels) == len(ensemble_logits), f"VAL! len(ensamble_labels) must be equal to len(ensamble_logits). Found {len(ensemble_labels)} and {len(ensemble_logits)}"

            # Compute loss and accuracy across ensemble
            batch_loss = 0
            batch_accuracy = 0

            for labels, logits in zip(ensemble_labels, ensemble_logits):
                # labels: (batch_size, seq_len), logits: (batch_size, seq_len, vocab_size)
                logits_transposed = logits.transpose(1, 2)
                batch_loss += F.cross_entropy(logits_transposed, labels)
                batch_accuracy += (logits_transposed.argmax(dim=1) == labels).float().mean()
            
            batch_accuracy /= len(ensemble_logits)
            
            val_losses.append(batch_loss.item())
            val_accuracies.append(batch_accuracy.item())
    
    return np.mean(val_losses), np.mean(val_accuracies)