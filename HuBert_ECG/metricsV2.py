import torch
import torch.nn as nn

from rich.console import Console
from rich.table import Table
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


def check_label_distribution(labels):
    """
    Check which labels appear in the dataset.
    
    Args:
        labels: Tensor of shape (num_samples, num_labels) for multilabel classification
                Each row is one sample, each column is one label (0 or 1)
    """
    # Sum along the sample dimension (dim=0) to count positives for each label
    label_counts = labels.sum(dim=0)
    print(f"Label distribution (# of positive samples per label): {label_counts}")
    print(f"Missing labels (count=0): {(label_counts == 0).nonzero().squeeze()}")


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


    def to(self, device):
        """Move all metrics to the specified device."""
        super().to(device)
        for metric in self.metrics.values():
            metric.to(device)
        return self
    

    def reset(self):
        """Reset all accumulated states for a new computation cycle."""
        self.loss_sum = 0.0
        self.num_batches = 0
        
        # Reset all metrics
        for metric in self.metrics.values():
            metric.reset()


    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: Optional[torch.Tensor] = None) -> None:
        """Update states with predictions and targets from a new batch."""
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
        
        # Print metrics using rich table
        self.print_metrics_table(results)
        
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


    def print_metrics_table(self, results: Dict[str, float]) -> None:
        """
        Print metrics in a formatted table using the rich library.
        
        Args:
            results: Dictionary containing computed metrics
        """
        console = Console()
        
        # Create table with title
        table = Table(title=f"{self.split.capitalize()} Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")
        
        # Add loss first
        loss_key = f"{self.split}_loss"
        if loss_key in results:
            table.add_row("Loss", f"{results[loss_key]:.4f}")
        
        # Add macro averages for each metric
        for name in self.metrics.keys():
            macro_key = f"{self.split}_{name}_macro"
            scalar_key = f"{self.split}_{name}"
            
            if macro_key in results:
                table.add_row(f"{name.capitalize()} (macro)", f"{results[macro_key]:.4f}")
            elif scalar_key in results:
                table.add_row(name.capitalize(), f"{results[scalar_key]:.4f}")
        
        console.print(table)
