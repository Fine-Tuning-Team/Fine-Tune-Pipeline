# MLflow Reporter Integration Guide

This guide explains how to integrate the MLflow reporter with the fine-tuning pipeline components.

## Overview

The `MLFlowReporter` class provides comprehensive tracking for:
- **FineTuner**: Configuration, training metrics, and model artifacts
- **Inferencer**: Configuration, performance metrics, and inference results  
- **Evaluator**: Configuration, evaluation metrics, and evaluation reports

## Configuration

Add MLflow settings to your `config.toml`:

```toml
[mlflow]
tracking_uri = "http://0.0.0.1:5000"  # MLflow tracking server URI 
experiment_name = "fine-tuning-experiment"  # MLflow experiment name
run_name = "null"  # Leave null for random name
run_name_prefix = ""
run_name_suffix = ""
```

## Basic Usage

### 1. Standalone Component Tracking

```python
from mlflow_reporter import MLFlowReporter
from finetuner import FineTune

# Initialize
mlflow_reporter = MLFlowReporter()
finetuner = FineTune()

# Track fine-tuning with context manager
with mlflow_reporter:
    # Log configuration
    mlflow_reporter.log_finetuner_config(finetuner.config)
    
    # Run and track
    trainer_stats = finetuner.run()
    mlflow_reporter.log_finetuner_metrics(trainer_stats)
    mlflow_reporter.log_finetuner_artifacts()
```

### 2. Manual Run Management

```python
mlflow_reporter = MLFlowReporter()

# Start run
mlflow_reporter.start_run("my-experiment-run")

# Log data
mlflow_reporter.log_param("learning_rate", 0.001)
mlflow_reporter.log_metric("accuracy", 0.95)

# End run
mlflow_reporter.end_run()
```

## Integration with Each Component

### FineTuner Integration

```python
def run_finetuning_with_mlflow():
    mlflow_reporter = MLFlowReporter()
    finetuner = FineTune()
    
    with mlflow_reporter:
        # Log configuration parameters
        mlflow_reporter.log_finetuner_config(finetuner.config)
        
        # Log environment info
        mlflow_reporter.log_environment_info()
        
        # Run training
        trainer_stats = finetuner.run()
        
        # Log training metrics and artifacts
        mlflow_reporter.log_finetuner_metrics(trainer_stats)
        mlflow_reporter.log_finetuner_artifacts()
```

**Logged Data:**
- Configuration: model ID, LoRA parameters, training settings
- Metrics: train/eval loss, learning rate, training speed
- Artifacts: fine-tuned model files, tokenizer

### Inferencer Integration

```python
def run_inference_with_mlflow():
    mlflow_reporter = MLFlowReporter()
    inferencer = Inferencer()
    
    with mlflow_reporter:
        # Log configuration
        mlflow_reporter.log_inferencer_config(inferencer.config)
        
        # Time the inference
        import time
        start_time = time.time()
        inferencer.run()
        inference_time = time.time() - start_time
        
        # Log performance metrics
        dataset_size = len(testing_dataset)  # Get actual size
        mlflow_reporter.log_inferencer_metrics(dataset_size, inference_time)
        
        # Log output artifacts
        mlflow_reporter.log_inferencer_artifacts()
```

**Logged Data:**
- Configuration: model settings, generation parameters
- Metrics: dataset size, inference time, samples per second
- Artifacts: inference results (JSONL file)

### Evaluator Integration

```python
def run_evaluation_with_mlflow():
    mlflow_reporter = MLFlowReporter()
    evaluator = Evaluator()
    
    with mlflow_reporter:
        # Log configuration
        mlflow_reporter.log_evaluator_config(evaluator.config)
        
        # Run evaluation
        evaluator.run()
        
        # Log evaluation metrics
        mlflow_reporter.log_evaluator_metrics(evaluator.evaluation_results)
        
        # Log evaluation artifacts
        mlflow_reporter.log_evaluator_artifacts()
```

**Logged Data:**
- Configuration: metrics used, LLM/embedding models
- Metrics: RAGAS scores (BLEU, ROUGE, semantic similarity, etc.)
- Artifacts: detailed results (Excel), summary (JSON)

## Full Pipeline Tracking

```python
def run_full_pipeline():
    mlflow_reporter = MLFlowReporter()
    
    # Main pipeline run
    with mlflow_reporter:
        mlflow_reporter.log_environment_info()
        
        # Stage 1: Fine-tuning (nested run)
        with mlflow_reporter.create_nested_run("finetuning"):
            finetuner = FineTune()
            mlflow_reporter.log_finetuner_config(finetuner.config)
            trainer_stats = finetuner.run()
            mlflow_reporter.log_finetuner_metrics(trainer_stats)
            mlflow_reporter.log_finetuner_artifacts()
        
        # Stage 2: Inference (nested run)
        with mlflow_reporter.create_nested_run("inference"):
            inferencer = Inferencer()
            mlflow_reporter.log_inferencer_config(inferencer.config)
            inferencer.run()
            mlflow_reporter.log_inferencer_artifacts()
        
        # Stage 3: Evaluation (nested run)
        with mlflow_reporter.create_nested_run("evaluation"):
            evaluator = Evaluator()
            mlflow_reporter.log_evaluator_config(evaluator.config)
            evaluator.run()
            mlflow_reporter.log_evaluator_metrics(evaluator.evaluation_results)
            mlflow_reporter.log_evaluator_artifacts()
```

## Key Features

### 🎯 **Automatic Configuration Logging**
- Reads from `config.toml` automatically
- Logs all relevant parameters for each component
- Handles type conversion and null values

### 📊 **Comprehensive Metrics Tracking**
- Training metrics: loss, learning rate, speed
- Inference metrics: dataset size, processing time
- Evaluation metrics: all RAGAS scores

### 📁 **Artifact Management**
- Model files and tokenizers
- Inference results and evaluation reports
- Automatic artifact organization

### 🔄 **Context Manager Support**
- Automatic run start/stop
- Error handling and logging
- Clean resource management

### 🌳 **Nested Runs**
- Organize pipeline stages
- Compare different runs
- Hierarchical experiment tracking

## MLflow UI Access

After running experiments, access the MLflow UI:

```bash
# Start MLflow server (if not already running)
mlflow server --host 0.0.0.0 --port 5000

# Open browser to: http://localhost:5000
```

## Best Practices

1. **Use Context Managers**: Always use `with mlflow_reporter:` for automatic cleanup
2. **Nested Runs for Pipelines**: Use nested runs to organize complex workflows
3. **Environment Logging**: Log environment info for reproducibility
4. **Error Handling**: MLflow reporter automatically logs errors in context manager
5. **Consistent Naming**: Use consistent run names and prefixes for easy organization

## Troubleshooting

### Connection Issues
- Ensure MLflow server is running on specified URI
- Check firewall/network settings

### Permission Issues  
- Verify write permissions for artifact storage
- Check MLflow server authentication settings

### Missing Metrics
- Ensure trainer_stats object has log_history attribute
- Verify evaluation_results object structure

## Environment Variables

Optional environment variables for authentication:
- `MLFLOW_USERNAME`: MLflow server username
- `MLFLOW_PASSWORD`: MLflow server password
