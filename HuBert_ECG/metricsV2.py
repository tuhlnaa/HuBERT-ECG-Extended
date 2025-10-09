import torch
import torch.nn as nn

from typing import Dict, Optional
from torcheval.metrics import MultilabelAUPRC
from torchmetrics.classification import (
    MulticlassAUROC,
    MultilabelAUROC,
    MulticlassAccuracy,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelSpecificity,
)


class FinetuneMetrics(nn.Module):
    """A PyTorch module for computing task-specific metrics during finetuning."""

    def __init__(self, task: str, num_labels: int, split: str = 'val'):
        """
        Initialize the metrics tracker for finetuning.

        Args:
            task: Type of task ('multi_label', 'multi_class', or 'regression')
            num_labels: Number of labels/classes (vocab_size)
            split: The data split ('train', 'val', or 'test')
        """
        super().__init__()
        self.task = task
        self.num_labels = num_labels
        self.split = split

        # Initialize task-specific metrics
        self.metrics = self._initialize_metrics()
        
        # Reset to initialize states
        self.reset()


    def _initialize_metrics(self) -> Dict[str, nn.Module]:
        """Initialize metrics based on task type."""
        task2metric = {
            'multi_label': {
                "f1-score": MultilabelF1Score(num_labels=self.num_labels, average=None),
                "recall": MultilabelRecall(num_labels=self.num_labels, average=None),
                "specificity": MultilabelSpecificity(num_labels=self.num_labels, average=None),
                "precision": MultilabelPrecision(num_labels=self.num_labels, average=None),
                "auroc": MultilabelAUROC(num_labels=self.num_labels, average=None),
                "auprc": MultilabelAUPRC(num_labels=self.num_labels, average=None),
            },
            'multi_class': {
                'accuracy': MulticlassAccuracy(num_classes=self.num_labels),
                'auroc': MulticlassAUROC(num_classes=self.num_labels)
            },
            'regression': {}
        }
        
        return task2metric[self.task]


    def reset(self):
        """Reset all accumulated states for a new computation cycle."""
        self.loss_sum = 0.0
        self.num_batches = 0
        
        # Reset all metrics
        for metric in self.metrics.values():
            metric.reset()


    def to(self, device):
        """Move all metrics to the specified device."""
        super().to(device)
        for metric in self.metrics.values():
            metric.to(device)
        return self


    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: Optional[torch.Tensor] = None) -> None:
        """
        Update states with predictions and targets from a new batch.
        
        Args:
            logits: Model output logits
            labels: Ground truth labels
            loss: Loss value for the batch
        """
        # Update loss tracking if provided
        if loss is not None:
            self.loss_sum += loss.item()
        
        self.num_batches += 1
        
        # Convert labels for metrics (typically need long type)
        labels = labels.long()
        
        # Update all metrics with the batch
        for metric in self.metrics.values():
            metric.update(logits, labels)


    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics from accumulated data.
        
        Returns:
            Dictionary containing loss and all computed metrics (both per-class and macro averages)
        """
        if self.num_batches == 0:
            print(f"Warning: No data accumulated for {self.split} metrics")
            return {f"{self.split}_loss": 0.0}
        
        # Calculate mean loss
        mean_loss = self.loss_sum / self.num_batches
        
        # Create metrics dictionary starting with loss
        results = {f"{self.split}_loss": mean_loss}
        
        # Compute all metrics
        for name, metric in self.metrics.items():
            score = metric.compute()
            
            # Calculate macro average (mean across all labels/classes)
            if isinstance(score, torch.Tensor):
                # Filter out NaN values before computing mean
                valid_scores = score[~torch.isnan(score)]
                macro = valid_scores.mean().item() if len(valid_scores) > 0 else 0.0
                
                # Add macro average
                results[f"{self.split}_{name}_macro"] = macro
                
                # Add per-class/label metrics
                score_list = score.cpu().tolist()
                for i, class_score in enumerate(score_list):
                    results[f"{self.split}_{name}_class_{i}"] = float(class_score)
            else:
                # For scalar metrics (like accuracy)
                results[f"{self.split}_{name}"] = float(score)
        
        # Print metrics summary
        metric_summary = f"{self.split.capitalize()} metrics - Loss: {mean_loss:.4f}"
        for name in self.metrics.keys():
            if f"{self.split}_{name}_macro" in results:
                metric_summary += f", {name}: {results[f'{self.split}_{name}_macro']:.4f}"
            elif f"{self.split}_{name}" in results:
                metric_summary += f", {name}: {results[f'{self.split}_{name}']:.4f}"
        
        print(metric_summary)
        
        return results


    def get_target_metric(self, target_metric_name: str) -> float:
        """
        Get the macro average of a specific target metric.
        
        Args:
            target_metric_name: Name of the target metric to retrieve
            
        Returns:
            Macro average of the target metric
        """
        if target_metric_name not in self.metrics:
            raise ValueError(f"Target metric {target_metric_name} not available for task {self.task}")
        
        score = self.metrics[target_metric_name].compute()
        
        if isinstance(score, torch.Tensor):
            valid_scores = score[~torch.isnan(score)]
            return valid_scores.mean().item() if len(valid_scores) > 0 else 0.0
        else:
            return float(score)