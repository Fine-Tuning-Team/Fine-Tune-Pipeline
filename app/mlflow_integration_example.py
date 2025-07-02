#!/usr/bin/env python3
"""
Example integration of MLflow reporter with FineTuner, Inferencer, and Evaluator.
This file shows how to integrate the MLflow reporter with each component.

Note: As of the latest updates, MLflow logging is now directly integrated into each component.
This file serves as an example of how to use the components with MLflow tracking.
"""

import time
from mlflow_reporter import MLFlowReporter
from finetuner import FineTune
from inferencer import Inferencer
from evaluator import Evaluator


def run_finetuner_with_mlflow():
    """Run FineTuner with integrated MLflow logging."""
    print("🚀 Starting Fine-tuning with MLflow integration...")
    
    # FineTuner now has MLflow integration built-in
    finetuner = FineTune()
    
    try:
        # MLflow logging is handled automatically within the run method
        trainer_stats = finetuner.run()
        print("✅ Fine-tuning completed successfully with MLflow logging!")
        return trainer_stats
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        raise


def run_inferencer_with_mlflow():
    """Run Inferencer with integrated MLflow logging."""
    print("🚀 Starting Inference with MLflow integration...")
    
    # Inferencer now has MLflow integration built-in
    inferencer = Inferencer()
    
    try:
        # MLflow logging is handled automatically within the run method
        inferencer.run()
        print("✅ Inference completed successfully with MLflow logging!")
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        raise


def run_evaluator_with_mlflow():
    """Run Evaluator with integrated MLflow logging."""
    print("🚀 Starting Evaluation with MLflow integration...")
    
    # Evaluator now has MLflow integration built-in
    evaluator = Evaluator()
    
    try:
        # MLflow logging is handled automatically within the run method
        evaluator.run()
        print("✅ Evaluation completed successfully with MLflow logging!")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise

def run_full_pipeline_with_mlflow():
    """
    Run the full pipeline with integrated MLflow tracking.
    Each component now has built-in MLflow integration.
    """
    print("🚀 Starting full pipeline with integrated MLflow tracking...")
    
    try:
        # Stage 1: Fine-tuning
        print("\n📋 Stage 1: Fine-tuning")
        run_finetuner_with_mlflow()
        
        # Stage 2: Inference
        print("\n📋 Stage 2: Inference")
        run_inferencer_with_mlflow()
        
        # Stage 3: Evaluation
        print("\n📋 Stage 3: Evaluation")
        run_evaluator_with_mlflow()
        
        print("\n🎉 Full pipeline completed successfully with MLflow tracking!")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise


def run_pipeline_with_custom_mlflow_tracking():
    """
    Example of running the pipeline with additional custom MLflow tracking.
    This shows how you can add extra MLflow logging on top of the built-in integration.
    """
    mlflow_reporter = MLFlowReporter()
    
    with mlflow_reporter:
        mlflow_reporter.log_environment_info()
        mlflow_reporter.log_pipeline_info("full_pipeline", "started", "Starting full pipeline with custom tracking")
        
        try:
            # Stage 1: Fine-tuning (with nested run)
            with mlflow_reporter.create_nested_run("finetuning_stage"):
                finetuner = FineTune()
                # Additional custom logging before running
                mlflow_reporter.log_param("custom_pipeline_stage", "finetuning")
                trainer_stats = finetuner.run()
                # Additional custom logging after running
                mlflow_reporter.log_param("finetuning_completed", True)
            
            # Stage 2: Inference (with nested run)
            with mlflow_reporter.create_nested_run("inference_stage"):
                inferencer = Inferencer()
                mlflow_reporter.log_param("custom_pipeline_stage", "inference")
                inferencer.run()
                mlflow_reporter.log_param("inference_completed", True)
            
            # Stage 3: Evaluation (with nested run)
            with mlflow_reporter.create_nested_run("evaluation_stage"):
                evaluator = Evaluator()
                mlflow_reporter.log_param("custom_pipeline_stage", "evaluation")
                evaluator.run()
                mlflow_reporter.log_param("evaluation_completed", True)
            
            # Log overall pipeline success
            mlflow_reporter.log_pipeline_info("full_pipeline", "completed", "Full pipeline completed successfully")
            
        except Exception as e:
            mlflow_reporter.log_pipeline_info("full_pipeline", "failed", str(e))
            raise


if __name__ == "__main__":
    print("🔧 MLflow Integration Examples")
    print("=" * 50)
    
    # Example usage - uncomment the function you want to run
    
    # Run individual components with built-in MLflow integration
    # run_finetuner_with_mlflow()
    # run_inferencer_with_mlflow()
    # run_evaluator_with_mlflow()
    
    # Run the full pipeline with built-in MLflow integration
    # run_full_pipeline_with_mlflow()
    
    # Run the pipeline with additional custom MLflow tracking
    # run_pipeline_with_custom_mlflow_tracking()
    
    print("\n📝 Note: Each component now has built-in MLflow integration.")
    print("   Simply run the components normally and MLflow logging will be automatic!")
    print("\n✅ MLflow integration examples completed!")