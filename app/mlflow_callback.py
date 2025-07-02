"""
MLflow callback for Transformers Trainer integration.
This callback automatically logs training metrics to MLflow during fine-tuning.
"""

import mlflow
from transformers import TrainerCallback
from typing import Dict, Any


class MLflowCallback(TrainerCallback):
    """
    A custom callback for Transformers Trainer that logs metrics to MLflow.
    This callback will automatically log training metrics during the fine-tuning process.
    """
    
    def __init__(self, mlflow_reporter=None, log_model_checkpoints=False):
        """
        Initialize the MLflow callback.
        
        Args:
            mlflow_reporter: MLFlowReporter instance for logging
            log_model_checkpoints: Whether to log model checkpoints as artifacts
        """
        self.mlflow_reporter = mlflow_reporter
        self.log_model_checkpoints = log_model_checkpoints
        self.logged_metrics = set()
    
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        """
        Called when logging occurs during training.
        Logs metrics to MLflow in real-time.
        """
        if logs is None:
            return
            
        step = state.global_step
        
        # Log training metrics
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                try:
                    mlflow.log_metric(key, value, step=step)
                    self.logged_metrics.add(key)
                except Exception as e:
                    print(f"Warning: Failed to log metric {key}: {e}")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch."""
        try:
            mlflow.log_metric("epoch", state.epoch, step=state.global_step)
        except Exception as e:
            print(f"Warning: Failed to log epoch: {e}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        try:
            # Log additional epoch-level metrics
            mlflow.log_metric("epoch_completed", state.epoch, step=state.global_step)
        except Exception as e:
            print(f"Warning: Failed to log epoch completion: {e}")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        try:
            # Log training configuration
            mlflow.log_param("num_train_epochs", args.num_train_epochs)
            mlflow.log_param("per_device_train_batch_size", args.per_device_train_batch_size)
            mlflow.log_param("gradient_accumulation_steps", args.gradient_accumulation_steps)
            mlflow.log_param("learning_rate", args.learning_rate)
            mlflow.log_param("warmup_steps", args.warmup_steps)
            mlflow.log_param("weight_decay", args.weight_decay)
            mlflow.log_param("logging_steps", args.logging_steps)
            mlflow.log_param("save_steps", args.save_steps)
            mlflow.log_param("eval_steps", getattr(args, 'eval_steps', 'N/A'))
        except Exception as e:
            print(f"Warning: Failed to log training parameters: {e}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        try:
            # Log final training statistics
            mlflow.log_metric("total_training_steps", state.global_step)
            mlflow.log_metric("final_epoch", state.epoch)
            
            if hasattr(state, 'log_history') and state.log_history:
                # Log best metrics if available
                best_metrics = self._extract_best_metrics(state.log_history)
                for metric_name, metric_value in best_metrics.items():
                    mlflow.log_metric(f"best_{metric_name}", metric_value)
        except Exception as e:
            print(f"Warning: Failed to log final training statistics: {e}")
    
    def on_save(self, args, state, control, **kwargs):
        """Called when a model checkpoint is saved."""
        if self.log_model_checkpoints:
            try:
                checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
                if hasattr(mlflow, 'log_artifacts') and checkpoint_dir:
                    mlflow.log_artifacts(checkpoint_dir, f"checkpoints/checkpoint-{state.global_step}")
            except Exception as e:
                print(f"Warning: Failed to log checkpoint: {e}")
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Called after evaluation."""
        try:
            mlflow.log_metric("evaluation_completed", 1, step=state.global_step)
        except Exception as e:
            print(f"Warning: Failed to log evaluation completion: {e}")
    
    def _extract_best_metrics(self, log_history):
        """Extract best metrics from training log history."""
        best_metrics = {}
        
        for log_entry in log_history:
            for key, value in log_entry.items():
                if isinstance(value, (int, float)) and key.endswith('_loss'):
                    # For loss metrics, we want the minimum value
                    if key not in best_metrics or value < best_metrics[key]:
                        best_metrics[key] = value
                elif isinstance(value, (int, float)) and any(metric in key for metric in ['accuracy', 'f1', 'bleu', 'rouge']):
                    # For performance metrics, we want the maximum value
                    if key not in best_metrics or value > best_metrics[key]:
                        best_metrics[key] = value
        
        return best_metrics
