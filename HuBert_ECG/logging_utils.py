import json

from clearml import Task
from pathlib import Path
from typing import Dict, Union, Optional


class ClearMLLogger():
    """ClearML logging implementation."""
    
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
            auto_connect_frameworks={'pytorch': False}
        )
        self.task.set_initial_iteration(offset=0)

        # Store the logger for easy access
        self.logger = self.task.get_logger()

        # Force iteration-based reporting with a dummy metric
        # Do this early in your script, before any time-consuming operations
        self.logger.report_scalar(
            title="dummy", 
            series="force_iteration_reporting", 
            iteration=0, 
            value=0.0
        )

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def log_metrics(self, metrics: Dict[str, float], step: int, mode: str = 'train') -> None:
        """
        Log metrics to ClearML with organized grouping.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
            mode: Mode of operation ('train' or 'val')
        """
        for name, value in metrics.items():
            # Handle learning rate specially
            if name == 'learning_rate':
                self.logger.report_scalar(
                    title="Learning Rate", 
                    series="LR", 
                    value=value,
                    iteration=step
                )
                continue
                
            # Split into appropriate paths based on metric name
            if name.startswith(('train_', 'val_')):
                phase = 'Training' if name.startswith('train_') else 'Validation'
                metric_name = name.replace('train_', '').replace('val_', '')

                if metric_name == 'loss':
                    title = metric_name
                    series = phase
                elif '_macro' in metric_name:
                    title = metric_name.replace('_macro', '')  # e.g. f1-score
                    series = phase  # e.g. "Validation"
                elif "class" in metric_name:
                    title = name.rsplit('_', 1)[0]  # e.g. "val_f1-score_class"
                    series = metric_name.split('_', 1)[1]   # e.g. "class_0"
                else:
                    continue

                # elif metric_name == "grad_clip_ratio" or metric_name == "exploding_grad_ratio" or metric_name == "vanishing_grad_ratio":
                #     title = "grad_ratio"
                #     series = metric_name
                # elif "grad" in metric_name:
                #     title = "grad"
                #     series = metric_name
                    
                self.logger.report_scalar(
                    title=title,
                    series=series,
                    value=value,
                    iteration=step
                )
            else:
                pass

                # # If no prefix, use the provided mode
                # title = 'Training' if mode == 'train' else 'Validation'
                # self.logger.report_scalar(
                #     title=title,
                #     series=name,
                #     value=value,
                #     iteration=step
                # )


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
