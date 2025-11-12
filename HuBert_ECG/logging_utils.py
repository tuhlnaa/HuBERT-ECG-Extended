import json

from clearml import Task
from pathlib import Path
from typing import Dict, Union, Optional, Tuple


class MetricHandler:
    """Base class for metric handling strategies."""
    
    def can_handle(self, metric_name: str) -> bool:
        """Check if this handler can process the metric."""
        raise NotImplementedError
    
    def parse(self, metric_name: str) -> Tuple[str, str]:
        """Parse metric name into (title, series)."""
        raise NotImplementedError


class LearningRateHandler(MetricHandler):
    def can_handle(self, metric_name: str) -> bool:
        return metric_name == 'learning_rate'
    
    def parse(self, metric_name: str) -> Tuple[str, str]:
        return "Learning Rate", "LR"


class ClassMetricHandler(MetricHandler):
    """Handles per-class metrics like val_f1-score_class_0."""
    
    def can_handle(self, metric_name: str) -> bool:
        return 'class' in metric_name and metric_name.startswith(('train_', 'val_'))
    
    def parse(self, metric_name: str) -> Tuple[str, str]:
        # e.g., "val_f1-score_class_0" -> title="val_f1-score_class", series="class_0"
        title = metric_name.rsplit('_', 1)[0]
        series = metric_name.split('_', 1)[1]
        return title, series


class SimplePhasedMetricHandler(MetricHandler):
    """Handles metrics like train_loss, val_loss, train_sse, etc."""
    
    def __init__(self, metric_names: list):
        self.metric_names = metric_names
    
    def can_handle(self, metric_name: str) -> bool:
        for name in self.metric_names:
            if metric_name in [f'train_{name}', f'val_{name}']:
                return True
        return False
    
    def parse(self, metric_name: str) -> Tuple[str, str]:
        phase = 'Training' if metric_name.startswith('train_') else 'Validation'
        metric = metric_name.replace('train_', '').replace('val_', '')
        return metric, phase


class MacroMetricHandler(MetricHandler):
    """Handles metrics with _macro suffix."""
    
    def can_handle(self, metric_name: str) -> bool:
        return '_macro' in metric_name and metric_name.startswith(('train_', 'val_'))
    
    def parse(self, metric_name: str) -> Tuple[str, str]:
        phase = 'Training' if metric_name.startswith('train_') else 'Validation'
        metric = metric_name.replace('train_', '').replace('val_', '').replace('_macro', '')
        return metric, phase


class ClearMLLogger:
    """ClearML logging implementation with extensible metric handling."""
   
    def __init__(
        self,
        output_dir: Union[str, Path],
        project: str,
        task_name: Optional[str] = None,
        task_type: Optional[str] = "training",
        reuse_last_task_id: Optional[str] = None,
        tags: Optional[list] = None,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
       
        # Initialize ClearML task
        self.task = Task.init(
            project_name=project,
            task_name=task_name,
            task_type=task_type,
            continue_last_task=reuse_last_task_id,
            tags=tags,
            output_uri=True,
            auto_connect_frameworks={'pytorch': False, 'matplotlib': False}
        )
        self.task.set_initial_iteration(offset=0)

        self.logger = self.task.get_logger()

        # Force iteration-based reporting
        self.logger.report_scalar(
            title="dummy",
            series="force_iteration_reporting",
            iteration=0,
            value=0.0
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metric handlers (order matters - first match wins)
        self._metric_handlers = [
            LearningRateHandler(),
            # MacroMetricHandler(),
            # ClassMetricHandler(),
            # SimplePhasedMetricHandler(['loss', 'sse', 'db_score', 'ch_score']),
            # SimplePhasedMetricHandler(['grad_clip_ratio', 'exploding_grad_ratio', 'vanishing_grad_ratio']),
        ]


    def register_handler(self, handler: MetricHandler) -> None:
        """Add a custom metric handler."""
        self._metric_handlers.insert(0, handler)  # Add at beginning for priority


    def log_metrics(self, metrics: Dict[str, float], step: int, mode: str = 'train') -> None:
        """
        Log metrics to ClearML with organized grouping.
       
        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
            mode: Mode of operation ('train' or 'val')
        """
        for name, value in metrics.items():
            # Find appropriate handler
            for handler in self._metric_handlers:
                if handler.can_handle(name):
                    title, series = handler.parse(name)
                    self.logger.report_scalar(
                        title=title,
                        series=series,
                        value=value,
                        iteration=step
                    )
                    break
            # If no handler matched, silently skip (or log warning if desired)


    def log_args_as_json(self, args) -> None:
        """
        Log args as hyperparameters to ClearML.
        
        Args:
            args: Arguments object (typically from argparse) to log as hyperparameters
        """
        # Convert args to dictionary
        if hasattr(args, '__dict__'):
            args_dict = vars(args)
        else:
            args_dict = args
        
        # Save args to JSON file locally for backup
        args_file = self.output_dir / "args.json"
        with open(args_file, 'w') as f:
            json.dump(args_dict, f, indent=4, default=str)
        
        # Connect parameters to ClearML (preferred for hyperparameters)
        self.task.connect(args_dict, name='Args')

        # Upload to ClearML
        self.task.upload_artifact(name='args', artifact_object=args_dict)


    def log_artifact(self, local_path: str, artifact_path: str = None) -> None:
        """
        Log artifacts to ClearML.
        
        Args:
            local_path: Path to the file to upload
            artifact_path: Name for the artifact in ClearML
        """
        artifact_name = artifact_path or Path(local_path).name
        self.task.upload_artifact(artifact_name, local_path)
