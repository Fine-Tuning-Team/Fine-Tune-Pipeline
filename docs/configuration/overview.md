# Configuration Overview

The Fine-Tune Pipeline uses a TOML-based configuration system that allows you to customize every aspect of the training, inference, and evaluation processes. All configuration is centralized in the `config.toml` file.

## Configuration Structure

The configuration file is organized into five main sections:

```toml
[fine_tuner]     # Fine-tuning configuration
[inferencer]     # Inference configuration  
[evaluator]      # Evaluation configuration
[mlflow]         # MLflow experiment tracking
[pipeline]       # Pipeline orchestration settings
```

## Configuration Loading

The pipeline automatically loads configuration from `config.toml` in the project root. The configuration system supports:

- **Environment Variable Substitution**: Use `${VARIABLE_NAME}` syntax
- **Null Values**: Use `"null"` string for optional parameters
- **Type Conversion**: Automatic conversion of strings to appropriate types

### Example with Environment Variables

```toml
[fine_tuner]
base_model_id = "${BASE_MODEL_ID}"
wandb_project_name = "${WANDB_PROJECT}"
```

## Common Configuration Patterns

### Development vs Production

=== "Development Config"
    ```toml
    [mlflow]
    tracking_uri = "http://localhost:5000"
    experiment_name = "dev-fine-tune-pipeline"

    [pipeline]
    enable_finetuning = true
    enable_inference = true
    enable_evaluation = true
    
    [fine_tuner]
    base_model_id = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
    epochs = 1
    device_train_batch_size = 2
    push_to_hub = false
    ```

=== "Production Config"
    ```toml
    [mlflow]
    tracking_uri = "https://mlflow.example.com"
    experiment_name = "prod-fine-tune-pipeline"

    [pipeline]
    enable_finetuning = true
    enable_inference = true
    enable_evaluation = true
    
    [fine_tuner]
    base_model_id = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    epochs = 20
    device_train_batch_size = 8
    push_to_hub = true
    ```

### Memory-Constrained Training

For systems with limited GPU memory:

```toml
[fine_tuner]
load_in_4bit = true
device_train_batch_size = 1
grad_accumulation = 16
max_sequence_length = 1024
use_gradient_checkpointing = "unsloth"
```

### High-Performance Training

For systems with ample resources:

```toml
[fine_tuner]
load_in_4bit = false
device_train_batch_size = 16
grad_accumulation = 2
max_sequence_length = 4096
packing = true
```

## Configuration Validation

The pipeline validates configuration at startup and will report errors for:

- Missing required fields
- Invalid data types
- Warnings on additional fields not used by the pipeline

## Environment-Specific Configurations

### Using Multiple Config Files

You can override the default config file:

```python
from app.config_manager import ConfigManager

# Load custom config
config_manager = ConfigManager("configs/production.toml")
```

### Configuration Inheritance

Create base configurations and extend them:

```toml
# base.toml
[fine_tuner]
base_model_id = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
epochs = 3
learning_rate = 0.0002

# Override specific values in config.toml
[fine_tuner]
epochs = 5  # Override base value
device_train_batch_size = 8  # Add new value
```

## Configuration Best Practices

### 1. Version Control

✅ **Do**: Commit base configuration files
❌ **Don't**: Commit files with sensitive API keys

### 2. Documentation

Always document your configuration choices:

```toml
[fine_tuner]
# Using smaller model for development speed
base_model_id = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"

# Optimized for RTX 3080 (10GB VRAM)
device_train_batch_size = 4
grad_accumulation = 4  # Effective batch size: 16
```

## Configuration Sections

| Section | Purpose | Required |
|---------|---------|----------|
| `[fine_tuner]` | Model training configuration | ✅ Yes |
| `[inferencer]` | Inference and prediction settings | ✅ Yes |
| `[evaluator]` | Evaluation metrics and settings | ✅ Yesl |

## Advanced Configuration

### Custom Data Processing

```toml
[fine_tuner]
# Custom column mappings
question_column = "input_text"
ground_truth_column = "target_text"
system_prompt_column = "system_message"

# Override system prompt for all examples
system_prompt_override_text = "You are a helpful assistant specializing in..."
```

### Training Optimization

```toml
[fine_tuner]
# LoRA configuration
rank = 64              # Higher rank = more parameters
lora_alpha = 32        # Scaling factor
lora_dropout = 0.05    # Dropout for regularization

# Training efficiency
packing = true         # Pack multiple sequences per batch
use_gradient_checkpointing = "unsloth"  # Save memory
dataset_num_proc = 8   # Parallel data processing
```

## Next Steps

- [Fine-Tuner Configuration](fine-tuner.md) - Detailed fine-tuning options
- [Inferencer Configuration](inferencer.md) - Inference settings
- [Evaluator Configuration](evaluator.md) - Evaluation metrics and setup
