# MLflow Integration Guide - UPDATED

This guide explains how MLflow is now integrated directly into the Fine-Tune Pipeline for comprehensive experiment tracking and monitoring.

## 🚀 Overview

MLflow integration has been built directly into all three main components of the pipeline:

1. **FineTuner** - Tracks training metrics, hyperparameters, and model artifacts
2. **Inferencer** - Tracks inference performance and generated outputs  
3. **Evaluator** - Tracks evaluation metrics and detailed results

## ✨ Key Features

### Automatic Tracking
- ✅ All components now include built-in MLflow tracking
- ✅ No additional setup required - just run the components normally
- ✅ Metrics, parameters, and artifacts are logged automatically

### Real-time Training Metrics
- ✅ Custom MLflow callback for Transformers Trainer
- ✅ Logs training loss, validation loss, learning rate, and other metrics in real-time
- ✅ Tracks training progress step-by-step during fine-tuning

### Comprehensive Logging

#### FineTuner Logs:
- **Parameters**: Model configuration, training hyperparameters, dataset info
- **Metrics**: Training/validation loss, learning rates, performance metrics, best metrics
- **Artifacts**: Fine-tuned model files, tokenizer, checkpoints
- **Real-time**: Step-by-step training progress via custom callback

#### Inferencer Logs:
- **Parameters**: Model configuration, inference settings
- **Metrics**: Dataset size, inference time, samples per second, progress tracking
- **Artifacts**: Generated responses (JSONL files)

#### Evaluator Logs:
- **Parameters**: Evaluation configuration, metrics used
- **Metrics**: All RAGAS evaluation scores (BLEU, ROUGE, semantic similarity, etc.)
- **Artifacts**: Detailed evaluation reports (Excel, JSON)

## 🔧 Usage

### Basic Usage (Automatic)
Simply run any component - MLflow tracking happens automatically:

```python
# Fine-tuning with automatic MLflow tracking
finetuner = FineTune()
finetuner.run()  # ✅ MLflow logging happens automatically

# Inference with automatic MLflow tracking  
inferencer = Inferencer()
inferencer.run()  # ✅ MLflow logging happens automatically

# Evaluation with automatic MLflow tracking
evaluator = Evaluator()
evaluator.run()  # ✅ MLflow logging happens automatically
```

### Pipeline Integration
For full pipeline runs with organized tracking:

```python
from mlflow_integration_example import run_full_pipeline_with_mlflow

# Run the complete pipeline with integrated MLflow tracking
run_full_pipeline_with_mlflow()
```

### Custom Additional Tracking
Add extra MLflow logging on top of built-in integration:

```python
from mlflow_reporter import MLFlowReporter

mlflow_reporter = MLFlowReporter()
with mlflow_reporter:
    # Your custom logging
    mlflow_reporter.log_param("custom_parameter", value)
    
    # Components create their own tracking automatically
    finetuner = FineTune()
    finetuner.run()
```

## ⚙️ Configuration

MLflow settings in your `config.toml`:

```toml
[mlflow]
tracking_uri = "http://localhost:5000"  # Your MLflow server
experiment_name = "fine-tune-experiments"
run_name = "my-experiment"
run_name_prefix = "ft"
run_name_suffix = "v1"
```

## 🔐 Environment Variables

```bash
# For MLflow server authentication (if required)
export MLFLOW_USERNAME="your-username"
export MLFLOW_PASSWORD="your-password"

# For other integrations
export WANDB_TOKEN="your-wandb-token"
export HF_TOKEN="your-huggingface-token"
export OPENAI_API_KEY="your-openai-key"
```

## 🏗️ Architecture

### MLflow Components

1. **MLFlowReporter** (`mlflow_reporter.py`)
   - Central MLflow integration class
   - Handles run management, logging, and artifact storage

2. **MLflowCallback** (`mlflow_callback.py`) - **NEW**
   - Custom Transformers callback for real-time training metrics
   - Automatically logs training progress to MLflow during fine-tuning
   - Integrated directly into the SFTTrainer

3. **Direct Integration** - **NEW**
   - All components (FineTuner, Inferencer, Evaluator) now have built-in MLflow
   - Automatic initialization and tracking
   - Error handling and cleanup

### Data Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  FineTuner  │───▶│  Inferencer  │───▶│  Evaluator  │
│   +MLflow   │    │   +MLflow    │    │   +MLflow   │
└─────────────┘    └──────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────┐
│               MLflow Tracking Server                │
│  • Real-time metrics   • Performance data           │
│  • Model artifacts     • Generated outputs          │
│  • Hyperparameters    • Evaluation scores           │
└─────────────────────────────────────────────────────┘
```

## 📊 Logged Metrics

### Training Metrics (FineTuner)
- `train_loss` - Training loss per step (real-time)
- `eval_loss` - Validation loss per step (real-time)
- `learning_rate` - Learning rate schedule (real-time)
- `train_runtime` - Training time
- `train_samples_per_second` - Training throughput
- `best_*` - Best achieved metrics
- `epoch` - Current epoch (real-time)

### Inference Metrics (Inferencer)
- `inference_dataset_size` - Number of samples processed
- `inference_total_time` - Total inference time
- `samples_per_second` - Inference throughput
- `processed_samples` - Progress tracking (periodic updates)

### Evaluation Metrics (Evaluator)
- `eval_*` - All RAGAS metric scores (individual metrics)
- `eval_*_mean/std/min/max` - Statistical summaries for array metrics
- `evaluation_dataset_size` - Number of samples evaluated
- `evaluation_time_seconds` - Evaluation duration
- `evaluation_samples_per_second` - Evaluation throughput

## 📁 Artifacts

### Model Artifacts
- Fine-tuned model files (./models/fine_tuned)
- Tokenizer files
- Model configuration
- Training checkpoints (optional)

### Output Artifacts
- Inference results (inferencer_output.jsonl)
- Evaluation reports (evaluator_output_detailed.xlsx, evaluator_output_summary.json)
- Configuration files
- Logs and metadata

## 🎯 Best Practices

1. **Run Naming**: Use descriptive run names with prefixes/suffixes
2. **Experiment Organization**: Group related runs in the same MLflow experiment
3. **Real-time Monitoring**: Use MLflow UI to watch training progress live
4. **Artifact Management**: Regularly clean up old artifacts to save storage
5. **Comparison**: Use MLflow's comparison features to analyze different runs
6. **Error Handling**: Components handle MLflow errors gracefully

## 🔧 Troubleshooting

### Common Issues

1. **MLflow Server Connection**
   ```bash
   # Check if MLflow server is running
   curl http://localhost:5000
   
   # Start MLflow server if needed
   mlflow server --host 0.0.0.0 --port 5000
   ```

2. **Authentication Issues**
   ```bash
   # Set authentication environment variables
   export MLFLOW_USERNAME="username"  
   export MLFLOW_PASSWORD="password"
   ```

3. **Storage Issues**
   - Ensure sufficient disk space for artifacts
   - Configure artifact storage location in MLflow

### Debugging

Enable debug logging:

```python
import logging
logging.getLogger("mlflow").setLevel(logging.DEBUG)
```

## 🔗 Integration with Other Tools

Works alongside existing integrations:
- **Weights & Biases**: Both WandB and MLflow run simultaneously
- **Hugging Face Hub**: Model pushing works with MLflow artifact logging  
- **Local Storage**: All artifacts saved locally AND logged to MLflow

## 🚀 Advanced Usage

### Custom Metrics in Components
```python
# Add this in your component code
if hasattr(self, 'mlflow_reporter') and self.mlflow_reporter:
    self.mlflow_reporter.log_metric("custom_metric", value)
```

### Nested Runs for Complex Experiments
```python
with mlflow_reporter.create_nested_run("experiment_phase"):
    # Run component or custom code
    pass
```

### Custom Artifact Logging
```python
mlflow_reporter.log_artifact("path/to/file.txt")
mlflow_reporter.log_artifacts("path/to/directory/")
```

## 📈 What's New

### Automatic Integration
- ✅ No more manual MLflow setup required
- ✅ Built into all components by default
- ✅ Automatic error handling and cleanup

### Real-time Training Tracking
- ✅ Custom MLflow callback for live training metrics
- ✅ Step-by-step loss tracking during fine-tuning
- ✅ Learning rate and performance monitoring

### Enhanced Metrics
- ✅ More comprehensive metric logging
- ✅ Statistical summaries for array-based metrics
- ✅ Progress tracking for long-running operations

### Better Organization
- ✅ Improved run naming and organization
- ✅ Environment info logging
- ✅ Pipeline stage tracking

This integration provides comprehensive, automatic experiment tracking across the entire fine-tuning pipeline with minimal setup required! 🎉
