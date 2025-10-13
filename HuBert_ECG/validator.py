import torch

from tqdm import tqdm
from typing import Dict, Any


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
