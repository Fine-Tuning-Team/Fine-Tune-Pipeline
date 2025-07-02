import mlflow
from mlflow import MlflowClient
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

# Local imports
from config_manager import get_config_manager, MLFlowConfig
from utils import setup_run_name


class MLFlowReporter:
    """
    MLflow reporter for tracking experiments across the fine-tuning pipeline.
    Integrates with FineTuner, Inferencer, and Evaluator components.
    """
    
    def __init__(self, config_manager=None):
        if config_manager is None:
            config_manager = get_config_manager()
        
        self.config = MLFlowConfig.from_config(config_manager)
        self.client = MlflowClient(tracking_uri=self.config.tracking_uri)
        
        # Set tracking URI and experiment
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)
        
        self.run_name = None
        self.current_run = None
        
    def setup_run_name(self):
        """Setup a run name for the MLflow experiment."""
        if self.run_name is None:
            self.run_name = setup_run_name(
                name=self.config.run_name,
                prefix=self.config.run_name_prefix,
                suffix=self.config.run_name_suffix,
            )
        return self.run_name

    def start_run(self, run_name: str | None = None, nested: bool = False):
        """Start an MLflow run."""
        if run_name is None:
            run_name = self.setup_run_name()
        
        self.current_run = mlflow.start_run(run_name=run_name, nested=nested)
        return self.current_run

    def end_run(self):
        """End the current MLflow run."""
        if self.current_run is not None:
            mlflow.end_run()
            self.current_run = None

    def log_metric(self, key: str, value: float, step: int = 0):
        """Log a metric to MLflow."""
        mlflow.log_metric(key, value, step=step)

    def log_param(self, key: str, value: Any):
        """Log a parameter to MLflow."""
        mlflow.log_param(key, str(value))

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact to MLflow."""
        mlflow.log_artifact(local_path, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log multiple artifacts from a directory to MLflow."""
        mlflow.log_artifacts(local_dir, artifact_path)

    def log_model_info(self, model_path: str, artifact_path: str = "model"):
        """Log model information as artifact."""
        if os.path.exists(model_path):
            self.log_artifacts(model_path, artifact_path)

    # ==================== FINETUNER INTEGRATION ====================
    
    def log_finetuner_config(self, config):
        """Log fine-tuner configuration parameters."""
        config_dict = {
            "base_model_id": config.base_model_id,
            "max_sequence_length": config.max_sequence_length,
            "dtype": str(config.dtype),
            "load_in_4bit": config.load_in_4bit,
            "load_in_8bit": config.load_in_8bit,
            "full_finetuning": config.full_finetuning,
            "rank": config.rank,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "target_modules": str(config.target_modules),
            "bias": config.bias,
            "training_data_id": config.training_data_id,
            "validation_data_id": str(config.validation_data_id),
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "device_train_batch_size": config.device_train_batch_size,
            "device_validation_batch_size": config.device_validation_batch_size,
            "grad_accumulation": config.grad_accumulation,
            "optimizer": config.optimizer,
            "weight_decay": config.weight_decay,
            "lr_scheduler_type": config.lr_scheduler_type,
            "seed": config.seed,
            "warmup_steps": config.warmup_steps,
            "packing": config.packing,
            "use_gradient_checkpointing": str(config.use_gradient_checkpointing),
            "train_on_responses_only": config.train_on_responses_only,
        }
        
        for key, value in config_dict.items():
            self.log_param(f"finetuner_{key}", value)

    def log_finetuner_metrics(self, trainer_stats):
        """Log fine-tuning training metrics."""
        if hasattr(trainer_stats, 'log_history'):
            for log_entry in trainer_stats.log_history:
                step = log_entry.get('step', 0)
                
                # Log training metrics
                if 'train_loss' in log_entry:
                    self.log_metric("train_loss", log_entry['train_loss'], step)
                if 'train_runtime' in log_entry:
                    self.log_metric("train_runtime", log_entry['train_runtime'], step)
                if 'train_samples_per_second' in log_entry:
                    self.log_metric("train_samples_per_second", log_entry['train_samples_per_second'], step)
                if 'train_steps_per_second' in log_entry:
                    self.log_metric("train_steps_per_second", log_entry['train_steps_per_second'], step)
                
                # Log evaluation metrics
                if 'eval_loss' in log_entry:
                    self.log_metric("eval_loss", log_entry['eval_loss'], step)
                if 'eval_runtime' in log_entry:
                    self.log_metric("eval_runtime", log_entry['eval_runtime'], step)
                    
                # Log learning rate
                if 'learning_rate' in log_entry:
                    self.log_metric("learning_rate", log_entry['learning_rate'], step)

    def log_finetuner_artifacts(self, model_dir: str = "./models/fine_tuned"):
        """Log fine-tuning artifacts (model files, tokenizer, etc.)."""
        if os.path.exists(model_dir):
            self.log_artifacts(model_dir, "fine_tuned_model")

    # ==================== INFERENCER INTEGRATION ====================
    
    def log_inferencer_config(self, config):
        """Log inferencer configuration parameters."""
        config_dict = {
            "testing_data_id": config.testing_data_id,
            "max_sequence_length": config.max_sequence_length,
            "dtype": str(config.dtype),
            "load_in_4bit": config.load_in_4bit,
            "load_in_8bit": config.load_in_8bit,
            "max_new_tokens": config.max_new_tokens,
            "use_cache": config.use_cache,
            "temperature": config.temperature,
            "min_p": config.min_p,
            "question_column": config.question_column,
            "ground_truth_column": config.ground_truth_column,
            "system_prompt_column": str(config.system_prompt_column),
            "system_prompt_override_text": str(config.system_prompt_override_text),
            "hf_user_id": config.hf_user_id,
        }
        
        for key, value in config_dict.items():
            self.log_param(f"inferencer_{key}", value)

    def log_inferencer_metrics(self, dataset_size: int, inference_time: Optional[float] = None):
        """Log inferencer performance metrics."""
        self.log_metric("inference_dataset_size", dataset_size)
        if inference_time:
            self.log_metric("inference_total_time", inference_time)
            self.log_metric("inference_samples_per_second", dataset_size / inference_time)

    def log_inferencer_artifacts(self, output_file: str = "inferencer_output.jsonl"):
        """Log inferencer output artifacts."""
        if os.path.exists(output_file):
            self.log_artifact(output_file, "inference_results")

    # ==================== EVALUATOR INTEGRATION ====================
    
    def log_evaluator_config(self, config):
        """Log evaluator configuration parameters."""
        config_dict = {
            "metrics": str(config.metrics),
            "llm": config.llm,
            "embedding": config.embedding,
        }
        
        for key, value in config_dict.items():
            self.log_param(f"evaluator_{key}", value)

    def log_evaluator_metrics(self, evaluation_results):
        """Log evaluation metrics from RAGAS evaluation."""
        if hasattr(evaluation_results, '_repr_dict'):
            results_dict = evaluation_results._repr_dict
            
            for metric_name, metric_value in results_dict.items():
                if isinstance(metric_value, (int, float)):
                    self.log_metric(f"eval_{metric_name}", metric_value)
                    
        # Also try to log individual metric values if available
        if hasattr(evaluation_results, 'values'):
            for key, value in evaluation_results.values.items():
                if isinstance(value, (int, float)):
                    self.log_metric(f"eval_{key}", value)
                elif hasattr(value, '__iter__') and not isinstance(value, str):
                    # Log aggregated statistics for array-like values
                    try:
                        import numpy as np
                        array_val = np.array(value)
                        if array_val.dtype.kind in 'bifc':  # numeric types
                            self.log_metric(f"eval_{key}_mean", float(np.mean(array_val)))
                            self.log_metric(f"eval_{key}_std", float(np.std(array_val)))
                            self.log_metric(f"eval_{key}_min", float(np.min(array_val)))
                            self.log_metric(f"eval_{key}_max", float(np.max(array_val)))
                    except (ImportError, Exception):
                        # If numpy is not available or conversion fails, skip
                        pass

    def log_evaluator_artifacts(self, 
                              summary_file: str = "evaluator_output_summary.json",
                              detailed_file: str = "evaluator_output_detailed.xlsx"):
        """Log evaluator output artifacts."""
        if os.path.exists(summary_file):
            self.log_artifact(summary_file, "evaluation_results")
        if os.path.exists(detailed_file):
            self.log_artifact(detailed_file, "evaluation_results")

    # ==================== PIPELINE INTEGRATION ====================
    
    def log_pipeline_info(self, stage: str, status: str, message: str = ""):
        """Log pipeline stage information."""
        self.log_param(f"pipeline_{stage}_status", status)
        if message:
            self.log_param(f"pipeline_{stage}_message", message)

    def create_nested_run(self, run_name: str):
        """Create a nested run for individual pipeline stages."""
        return mlflow.start_run(run_name=run_name, nested=True)

    def log_environment_info(self):
        """Log environment and system information."""
        import platform
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_version = getattr(torch, 'version', {}).get('cuda', 'N/A') if cuda_available else "N/A"
            gpu_count = torch.cuda.device_count() if cuda_available else 0
        except ImportError:
            cuda_available = False
            cuda_version = "N/A"
            gpu_count = 0
        
        env_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cuda_available": cuda_available,
            "cuda_version": cuda_version,
            "gpu_count": gpu_count,
        }
        
        for key, value in env_info.items():
            self.log_param(f"env_{key}", str(value))

    # ==================== CONTEXT MANAGERS ====================
    
    def __enter__(self):
        """Context manager support - start run."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support - end run."""
        if exc_type is not None:
            # Log error information if an exception occurred
            self.log_param("error_type", str(exc_type.__name__))
            self.log_param("error_message", str(exc_val))
        self.end_run()