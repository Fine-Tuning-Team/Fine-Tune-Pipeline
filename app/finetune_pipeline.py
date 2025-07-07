import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any

import mlflow
from mlflow import MlflowClient

# Local imports
from config_manager import get_config_manager, MLFlowConfig
from finetuner import FineTune
from inferencer import Inferencer
from evaluator import Evaluator
from utils import setup_run_name

PIPELINE_VERSION = "1.0.0"


class FineTunePipeline:
    """
    Pipeline orchestrator that runs finetuning, inference, and evaluation in sequence
    with comprehensive MLflow tracking.
    """
    
    def __init__(self, config_path: str = "config.toml", 
                 enable_finetuning: bool = True,
                 enable_inference: bool = True, 
                 enable_evaluation: bool = True,
                 stop_after_finetuning: bool = False,
                 stop_after_inference: bool = False):
        """Initialize the pipeline with configuration and phase control."""
        self.config_manager = get_config_manager(config_path)
        self.mlflow_config = MLFlowConfig.from_config(self.config_manager)
        
        # Pipeline components
        self.finetuner = None
        self.inferencer = None
        self.evaluator = None
        
        # MLflow tracking
        self.mlflow_client = None
        self.experiment_id = None
        self.run_id = None
        self.run_name = None
        
        # Pipeline state
        self.pipeline_start_time = None
        self.pipeline_end_time = None
        self.pipeline_results = {}
        
        # Pipeline control flags
        self.enable_finetuning = enable_finetuning
        self.enable_inference = enable_inference
        self.enable_evaluation = enable_evaluation
        self.stop_after_finetuning = stop_after_finetuning
        self.stop_after_inference = stop_after_inference
        
    def setup_mlflow(self) -> None:
        """Setup MLflow tracking server and experiment."""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
            mlflow.set_experiment(self.mlflow_config.experiment_name)
            self.mlflow_client = MlflowClient(self.mlflow_config.tracking_uri)
            
            # TODO: Need to look into this logic
            # Create or get experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.mlflow_config.experiment_name)
                if experiment is None:
                    self.experiment_id = mlflow.create_experiment(self.mlflow_config.experiment_name)
                else:
                    self.experiment_id = experiment.experiment_id
            except Exception:
                self.experiment_id = mlflow.create_experiment(self.mlflow_config.experiment_name)\
            # TODO: End of TODO
            
            # Setup run name
            self.run_name = setup_run_name(
                name=self.mlflow_config.run_name,
                prefix=self.mlflow_config.run_name_prefix,
                suffix=self.mlflow_config.run_name_suffix,
            )
            
            print(f"--- ✅ MLflow setup complete. Experiment: {self.mlflow_config.experiment_name}, Run: {self.run_name} ---")
            
        except Exception as e:
            print(f"--- ❌ MLflow setup failed: {e} ---")
            raise
    
    def start_mlflow_run(self) -> None:
        """Start MLflow run for the entire pipeline."""
        try:
            mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=self.run_name
            )
            active_run = mlflow.active_run()
            if active_run is not None:
                self.run_id = active_run.info.run_id
            else:
                raise RuntimeError("No active MLflow run found.")
            
            # Log pipeline start time
            self.pipeline_start_time = time.time()
            mlflow.log_param("pipeline_start_time", self.pipeline_start_time)
            mlflow.log_param("pipeline_version", PIPELINE_VERSION)
            
            print(f"--- ✅ MLflow run started: {self.run_id} ---")
            
        except Exception as e:
            print(f"--- ❌ Failed to start MLflow run: {e} ---")
            raise
    
    def log_configuration_to_mlflow(self) -> None:
        """Log all configuration parameters to MLflow."""
        try:
            # Get all config sections
            finetuner_config = self.config_manager.get_section("fine_tuner")
            inferencer_config = self.config_manager.get_section("inferencer")
            evaluator_config = self.config_manager.get_section("evaluator")
            mlflow_config = self.config_manager.get_section("mlflow")
            
            # Log finetuner parameters
            for key, value in finetuner_config.items():
                mlflow.log_param(f"finetuner_{key}", value)
            
            # Log inferencer parameters
            for key, value in inferencer_config.items():
                mlflow.log_param(f"inferencer_{key}", value)
            
            # Log evaluator parameters
            for key, value in evaluator_config.items():
                mlflow.log_param(f"evaluator_{key}", value)
            
            # Log MLflow config
            for key, value in mlflow_config.items():
                mlflow.log_param(f"mlflow_{key}", value)
            
            print("--- ✅ Configuration logged to MLflow ---")
            
        except Exception as e:
            print(f"--- ❌ Failed to log configuration: {e} ---")
            # Don't raise here, continue pipeline execution
    
    def run_finetuning(self) -> Dict[str, Any]:
        """Run the finetuning step and log results to MLflow."""
        print("\n=== STARTING FINETUNING PHASE ===")
        
        try:
            # Create nested run for finetuning
            with mlflow.start_run(nested=True, run_name=f"{self.run_name}_finetuning"):
                finetuning_start_time = time.time()
                mlflow.log_param("phase", "finetuning")
                
                # Initialize and run finetuner
                self.finetuner = FineTune(config_manager=self.config_manager)
                training_stats = self.finetuner.run()
                
                finetuning_end_time = time.time()
                finetuning_duration = finetuning_end_time - finetuning_start_time
                
                # Log finetuning metrics
                mlflow.log_metric("finetuning_duration_seconds", finetuning_duration)
                mlflow.log_metric("finetuning_duration_minutes", finetuning_duration / 60)
                
                # Log training statistics if available
                if training_stats:
                    # Log basic trainer stats
                    if hasattr(training_stats, 'training_loss'):
                        mlflow.log_metric("final_training_loss", training_stats.training_loss)
                    if hasattr(training_stats, 'eval_loss'):
                        mlflow.log_metric("final_eval_loss", training_stats.eval_loss)
                    if hasattr(training_stats, 'epoch'):
                        mlflow.log_metric("total_epochs", training_stats.epoch)
                    if hasattr(training_stats, 'global_step'):
                        mlflow.log_metric("total_training_steps", training_stats.global_step)

                
                # model_path = "./models/fine_tuned"
                # ===== DISABLED: As choroe only has in-memory storage =====
                # Log model artifacts
                # if Path(model_path).exists():
                #     mlflow.log_artifacts(model_path, "fine_tuned_model")
                # ====== END DISABLED =====
                
                finetuning_results = {
                    "status": "success",
                    "duration_seconds": finetuning_duration,
                    "training_stats": training_stats,
                    # "model_path": model_path
                }
                
                print(f"--- ✅ Finetuning completed in {finetuning_duration:.2f} seconds ---")
                return finetuning_results
                
        except Exception as e:
            error_msg = f"Finetuning failed: {e}"
            print(f"--- ❌ {error_msg} ---")
            mlflow.log_param("finetuning_error", str(e))
            return {"status": "failed", "error": error_msg}
    
    def run_inference(self) -> Dict[str, Any]:
        """Run the inference step and log results to MLflow."""
        print("\n=== STARTING INFERENCE PHASE ===")
        
        try:
            # Create nested run for inference
            with mlflow.start_run(nested=True, run_name=f"{self.run_name}_inference"):
                inference_start_time = time.time()
                mlflow.log_param("phase", "inference")
                
                # Initialize and run inferencer
                self.inferencer = Inferencer(config_manager=self.config_manager)
                self.inferencer.run()
                
                inference_end_time = time.time()
                inference_duration = inference_end_time - inference_start_time
                
                # Log inference metrics
                mlflow.log_metric("inference_duration_seconds", inference_duration)
                mlflow.log_metric("inference_duration_minutes", inference_duration / 60)
                
                # Log inference output as artifact
                output_file = "inferencer_output.jsonl"
                if Path(output_file).exists():
                    mlflow.log_artifact(output_file, "inference_outputs")
                    
                    # Count number of inferences
                    with open(output_file, 'r') as f:
                        inference_count = sum(1 for _ in f)
                    mlflow.log_metric("total_inferences", inference_count)
                
                inference_results = {
                    "status": "success",
                    "duration_seconds": inference_duration,
                    "output_file": output_file
                }
                
                print(f"--- ✅ Inference completed in {inference_duration:.2f} seconds ---")
                return inference_results
                
        except Exception as e:
            error_msg = f"Inference failed: {e}"
            print(f"--- ❌ {error_msg} ---")
            mlflow.log_param("inference_error", str(e))
            return {"status": "failed", "error": error_msg}
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run the evaluation step and log results to MLflow."""
        print("\n=== STARTING EVALUATION PHASE ===")
        
        try:
            # Create nested run for evaluation
            with mlflow.start_run(nested=True, run_name=f"{self.run_name}_evaluation"):
                evaluation_start_time = time.time()
                mlflow.log_param("phase", "evaluation")
                
                # Initialize and run evaluator
                self.evaluator = Evaluator(config_manager=self.config_manager)
                self.evaluator.run()
                
                evaluation_end_time = time.time()
                evaluation_duration = evaluation_end_time - evaluation_start_time
                
                # Log evaluation metrics
                mlflow.log_metric("evaluation_duration_seconds", evaluation_duration)
                mlflow.log_metric("evaluation_duration_minutes", evaluation_duration / 60)
                
                # Log evaluation results
                summary_results = self.evaluator.get_summary_results()
                for metric_name, metric_value in summary_results.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(f"eval_{metric_name}", metric_value)
                    else:
                        mlflow.log_param(f"eval_{metric_name}", str(metric_value))
                
                # Log evaluation artifacts
                detailed_file = "evaluator_output_detailed.xlsx"
                summary_file = "evaluator_output_summary.json"
                
                if Path(detailed_file).exists():
                    mlflow.log_artifact(detailed_file, "evaluation_outputs")
                if Path(summary_file).exists():
                    mlflow.log_artifact(summary_file, "evaluation_outputs")
                
                evaluation_results = {
                    "status": "success",
                    "duration_seconds": evaluation_duration,
                    "summary_results": summary_results,
                    "detailed_file": detailed_file,
                    "summary_file": summary_file
                }
                
                print(f"--- ✅ Evaluation completed in {evaluation_duration:.2f} seconds ---")
                return evaluation_results
                
        except Exception as e:
            error_msg = f"Evaluation failed: {e}"
            print(f"--- ❌ {error_msg} ---")
            mlflow.log_param("evaluation_error", str(e))
            return {"status": "failed", "error": error_msg}
    
    def stop_mlflow_run(self) -> None:
        """Finalize MLflow run with pipeline summary."""
        try:
            # Calculate total pipeline duration
            self.pipeline_end_time = time.time()
            if self.pipeline_start_time is None:
                self.pipeline_start_time = time.time()
            total_duration = self.pipeline_end_time - self.pipeline_start_time
            
            mlflow.log_metric("pipeline_total_duration_seconds", total_duration)
            mlflow.log_metric("pipeline_total_duration_minutes", total_duration / 60)
            mlflow.log_param("pipeline_end_time", self.pipeline_end_time)
            
            # Log pipeline success status
            all_successful = all(
                result.get("status") == "success" 
                for result in self.pipeline_results.values()
            )
            mlflow.log_param("pipeline_success", all_successful)
            
            # Log pipeline results summary
            results_summary = {
                "finetuning": self.pipeline_results.get("finetuning", {}).get("status", "not_run"),
                "inference": self.pipeline_results.get("inference", {}).get("status", "not_run"),
                "evaluation": self.pipeline_results.get("evaluation", {}).get("status", "not_run"),
                "total_duration_minutes": total_duration / 60
            }
            
            # Save results summary as artifact
            with open("pipeline_summary.json", "w") as f:
                json.dump(results_summary, f, indent=2)
            mlflow.log_artifact("pipeline_summary.json")
            
            print(f"--- ✅ Pipeline completed in {total_duration:.2f} seconds ---")
            print(f"--- MLflow Run ID: {self.run_id} ---")
            
        except Exception as e:
            print(f"--- ❌ Failed to finalize MLflow run: {e} ---")
        finally:
            mlflow.end_run()
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline: finetuning -> inference -> evaluation."""
        try:
            print("=== STARTING FINE-TUNE PIPELINE ===")
            
            # Setup MLflow
            self.setup_mlflow()
            self.start_mlflow_run()
            self.log_configuration_to_mlflow()
            
            # Run finetuning
            if self.enable_finetuning:
                self.pipeline_results["finetuning"] = self.run_finetuning()
                if self.pipeline_results["finetuning"]["status"] != "success":
                    print("--- ❌ Pipeline stopped due to finetuning failure ---")
                    return self.pipeline_results
                
                if self.stop_after_finetuning:
                    print("--- ⏹️ Pipeline stopped after finetuning as requested ---")
                    return self.pipeline_results
            else:
                print("--- ⏭️ Skipping finetuning phase ---")
                self.pipeline_results["finetuning"] = {"status": "skipped", "reason": "disabled"}
            
            # Run inference
            if self.enable_inference:
                self.pipeline_results["inference"] = self.run_inference()
                if self.pipeline_results["inference"]["status"] != "success":
                    print("--- ❌ Pipeline stopped due to inference failure ---")
                    return self.pipeline_results
                
                if self.stop_after_inference:
                    print("--- ⏹️ Pipeline stopped after inference as requested ---")
                    return self.pipeline_results
            else:
                print("--- ⏭️ Skipping inference phase ---")
                self.pipeline_results["inference"] = {"status": "skipped", "reason": "disabled"}
            
            # Run evaluation
            if self.enable_evaluation:
                self.pipeline_results["evaluation"] = self.run_evaluation()
            else:
                print("--- ⏭️ Skipping evaluation phase ---")
                self.pipeline_results["evaluation"] = {"status": "skipped", "reason": "disabled"}
            
            return self.pipeline_results
            
        except Exception as e:
            print(f"--- ❌ Pipeline failed with error: {e} ---")
            self.pipeline_results["pipeline_error"] = str(e)
            return self.pipeline_results
        
        finally:
            # Always finalize MLflow run
            self.stop_mlflow_run()


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="Run the Fine-Tune Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.toml",
        help="Path to the configuration file"
    )
    
    # Pipeline control arguments
    parser.add_argument(
        "--skip-finetuning",
        action="store_true", # If only the argument is provided, it will be set to True
        help="Skip the finetuning phase"
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip the inference phase"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip the evaluation phase"
    )
    parser.add_argument(
        "--stop-after-finetuning",
        action="store_true",
        help="Stop pipeline after finetuning phase completes"
    )
    parser.add_argument(
        "--stop-after-inference", 
        action="store_true",
        help="Stop pipeline after inference phase completes"
    )
    
    # API key arguments
    parser.add_argument(
        "--wandb-key", 
        type=str, 
        help="Weights & Biases API key"
    )
    parser.add_argument(
        "--hf-key", 
        type=str, 
        required=True,
        help="Hugging Face API token"
    )
    parser.add_argument(
        "--openai-key", 
        type=str, 
        required=True,
        help="OpenAI API key for LLM evaluation"
    )
    parser.add_argument(
        "--mlflow-username", 
        type=str, 
        help="MLflow username"
    )
    parser.add_argument(
        "--mlflow-password", 
        type=str, 
        help="MLflow password"
    )
    
    args = parser.parse_args()
    
    # Set environment variables from command-line arguments
    if args.wandb_key:
        os.environ["WANDB_TOKEN"] = args.wandb_key
    
    if args.hf_key:
        os.environ["HF_TOKEN"] = args.hf_key
    
    if args.openai_key:
        os.environ["OPENAI_API_KEY"] = args.openai_key
    
    if args.mlflow_username and args.mlflow_password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = args.mlflow_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = args.mlflow_password
    
    # Validate config file exists
    if not Path(args.config).exists():
        print(f"--- ❌ Configuration file not found: {args.config} ---")
        sys.exit(1)
    
    # Run the pipeline
    try:
        pipeline = FineTunePipeline(
            config_path=args.config,
            enable_finetuning=not args.skip_finetuning,
            enable_inference=not args.skip_inference,
            enable_evaluation=not args.skip_evaluation,
            stop_after_finetuning=args.stop_after_finetuning,
            stop_after_inference=args.stop_after_inference
        )
        results = pipeline.run_pipeline()
        
        # Print final summary
        print("\n=== PIPELINE SUMMARY ===")
        for phase, result in results.items():
            if isinstance(result, dict) and "status" in result:
                status_emoji = "✅" if result["status"] == "success" else "❌"
                print(f"{status_emoji} {phase.upper()}: {result['status']}")
                if "duration_seconds" in result:
                    print(f"   Duration: {result['duration_seconds']:.2f} seconds")
        
        # Exit with appropriate code
        all_successful = all(
            result.get("status") == "success" 
            for result in results.values() 
            if isinstance(result, dict) and "status" in result
        )
        sys.exit(0 if all_successful else 1)
        
    except Exception as e:
        print(f"--- ❌ Pipeline execution failed: {e} ---")
        sys.exit(1)


if __name__ == "__main__":
    main()