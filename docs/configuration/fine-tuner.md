# Fine-Tuner Configuration

The `[fine_tuner]` section controls all aspects of the model fine-tuning process. This page provides comprehensive documentation of all available configuration options.

## Model Configuration

### Base Model Settings

```toml
[fine_tuner]
# Required: Hugging Face model ID or local path
base_model_id = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"

# Maximum sequence length for training
max_sequence_length = 4096

# Data type for model weights (null for auto-detection)
dtype = "null"  # Options: "float16", "bfloat16", null

# Quantization settings (choose one)
load_in_4bit = true
load_in_8bit = false

# Whether to use full fine-tuning instead of LoRA
full_finetuning = false
```

### Model Resource Recommendations

| Recommended Model | Memory Requirement |
|-------------------|-------------------|
| `unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit` | 2GB |
| `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` | 4GB |
| `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` | 8GB |
| `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | 16GB |

## LoRA Configuration

Low-Rank Adaptation (LoRA) settings for parameter-efficient fine-tuning:

```toml
[fine_tuner]
# LoRA rank - higher values = more trainable parameters
rank = 16  # Typical values: 8, 16, 32, 64, ...

# LoRA alpha - scaling factor for LoRA updates
lora_alpha = 16  # Usually equal to rank

# Dropout rate for LoRA layers
lora_dropout = 0.1  # Typical Range: 0.0 - 0.3

# Target modules for LoRA adaptation
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
    "gate_proj", "up_proj", "down_proj"       # Feed-forward layers
]

# Bias handling
bias = "none"  # Options: "none", "all", "lora_only"

# Advanced LoRA options
use_rslora = false      # Rank-Stabilized LoRA
loftq_config = "null"   # LoFTQ configuration
```

### LoRA Performance Guide

| Rank | Parameters | Speed | Quality | Use Case |
|------|------------|-------|---------|----------|
| 8 | ~0.5M | Fast | Good | Quick prototyping |
| 16 | ~1M | Medium | Better | General purpose |
| 32 | ~2M | Slower | High | Quality-focused |
| 64 | ~4M | Slowest | Highest | Research/Production |

## Dataset Configuration

### Data Sources

```toml
[fine_tuner]
# Required: Training dataset
training_data_id = "your-huggingface-username/training-dataset"

# Optional: Validation dataset
validation_data_id = "your-huggingface-username/validation-dataset"  # or "null"

# Number of processes for dataset loading
dataset_num_proc = 4
```

### Column Mapping

Map your dataset columns to the expected format:

```toml
[fine_tuner]
# Required columns
question_column = "question"        # Input/instruction column
ground_truth_column = "answer"      # Target/response column

# Optional system prompt
system_prompt_column = "system"     # System prompt column (or "null")

# Override system prompt for all examples
system_prompt_override_text = "null"  # Custom system prompt (or "null")
```

### Dataset Format Examples

=== "Q&A Format"
    For domain adaptation tasks.
    ```json
    {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris.",
        "system": "You are a geography expert."
    }
    ```

=== "Instruction Format"
    For instruction-following tasks.
    ```json
    {
        "instruction": "Summarize the following text:",
        "input": "Long text to summarize...",
        "output": "Brief summary of the text."
    }
    ```

## Training Parameters

### Basic Training Settings

```toml
[fine_tuner]
# Number of training epochs
epochs = 30

# Learning rate
learning_rate = 0.0002  # Typical range: 1e-5 to 5e-4

# Batch sizes
device_train_batch_size = 4        # Per-device batch size
device_validation_batch_size = 4   # Validation batch size
grad_accumulation = 4              # Gradient accumulation steps

# Warmup and scheduling
warmup_steps = 5                   # Learning rate warmup
lr_scheduler_type = "linear"       # Options: "linear", "cosine", "constant"

# Optimization
optimizer = "paged_adamw_8bit"     # Memory-efficient optimizer
weight_decay = 0.01                # L2 regularization

# Random seed for reproducibility
seed = 42
```

### Advanced Training Options

```toml
[fine_tuner]
# Memory optimization
use_gradient_checkpointing = "unsloth"  # Options: true, false, "unsloth"
use_flash_attention = true              # Flash attention for efficiency
packing = false                         # Pack multiple sequences per batch

# Training on responses only
train_on_responses_only = true
question_part = "<|im_start|>user\n"      # Question template
answer_part = "<|im_start|>assistant\n"   # Answer template
```

!!! tip "Training on Responses Only"
    This option allows training the model on the responses directly, which results in higher accuracy. Typically the loss is calculated for the entire completion text, including the question and the system prompt that we provide. However, this option allows us to train the model on the model response section only, which results in higher accuracy.

## Logging and Monitoring

### Mlflow Integration

```toml
[fine_tuner]
# Logging configuration
log_steps = 10          # Log metrics every N steps
log_first_step = true   # Log the first step
report_to = "mlflow"     # Reporting backend: "wandb", "tensorboard", "mlflow", "none"

[mlflow]
# MLflow tracking URI and experiment settings
tracking_uri = "https://your-mlflow-tracking-uri"
experiment_name = "your-experiment-name"
run_name = "your-run-name"  # Custom run name or "null" for auto. Recommended to use a versioning scheme like "0.0.1"
```

### Model Saving

```toml
[fine_tuner]
# Checkpoint saving
save_steps = 20           # Save checkpoint every N steps
save_total_limit = 3      # Maximum number of checkpoints to keep (older ones will be deleted)

# Hugging Face Hub integration
push_to_hub = true        # Push final model to Hub
```

## Run Naming

Control how your training runs are named:

```toml
[mlflow]
# Run name configuration
run_name = "0.0.1"          # Custom run name (or "null" for auto)
run_name_prefix = ""        # Prefix for auto-generated names
run_name_suffix = ""        # Suffix for auto-generated names
```

!!! tip "Recommendation"
    Use a versioning scheme like `0.0.1` for `run_name` to easily track changes across runs. You can also use prefixes and suffixes to add context, e.g., prefix `exp-` for experiments or suffix `-alpha` for further versioning. (e.g., `exp-0.0.1-v1`).

### Run Name Examples

| Configuration | Generated Name |
|---------------|----------------|
| `run_name = "my-model"` | `my-model` |
| `run_name_prefix = "exp-"` | `exp-20250629-143022` |
| `run_name_suffix = "-v1"` | `20250629-143022-v1` |

## Memory Optimization Guide

### For 4GB GPU (RTX 3060)

```toml
[fine_tuner]
base_model_id = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
max_sequence_length = 1024
load_in_4bit = true
device_train_batch_size = 1
grad_accumulation = 16
use_gradient_checkpointing = "unsloth"
rank = 8
```

### For 8GB GPU (RTX 3070)

```toml
[fine_tuner]
base_model_id = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
max_sequence_length = 2048
load_in_4bit = true
device_train_batch_size = 2
grad_accumulation = 8
use_gradient_checkpointing = "unsloth"
rank = 16
```

### For 12GB+ GPU (RTX 3080 Ti/4070 Ti)

```toml
[fine_tuner]
base_model_id = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
max_sequence_length = 4096
load_in_4bit = true
device_train_batch_size = 4
grad_accumulation = 4
use_gradient_checkpointing = "unsloth"
rank = 32
```

## Performance Tuning

### Speed Optimization

```toml
[fine_tuner]
# Enable packing for 5x speed improvement on short sequences
packing = true

# Use flash attention
use_flash_attention = true

# Optimize data loading
dataset_num_proc = 8  # Match your CPU cores

# Efficient precision
dtype = "null"  # Auto-select best precision
```

### Quality Optimization

```toml
[fine_tuner]
# Higher LoRA rank for better quality
rank = 64
lora_alpha = 32

# More training epochs
epochs = 50

# Lower learning rate for stability
learning_rate = 0.0001

# Add validation dataset
validation_data_id = "your-username/validation-dataset"
```

## Common Configuration Patterns

### Research/Experimentation

```toml
[fine_tuner]
epochs = 1
device_train_batch_size = 1
push_to_hub = false
report_to = "none"
```

### Production Training

```toml
[fine_tuner]
epochs = 30
device_train_batch_size = 8
push_to_hub = true
save_steps = 100
save_total_limit = 5
```

### Memory-Constrained

```toml
[fine_tuner]
load_in_4bit = true
device_train_batch_size = 1
grad_accumulation = 32
use_gradient_checkpointing = "unsloth"
max_sequence_length = 1024
```

## Troubleshooting

### Out of Memory Errors

1. Reduce `device_train_batch_size`
2. Increase `grad_accumulation` to maintain effective batch size
3. Reduce `max_sequence_length`
4. Enable `use_gradient_checkpointing`
5. Use smaller model or higher quantization

### Slow Training

1. Enable `packing = true`
2. Enable `use_flash_attention = true`
3. Increase `dataset_num_proc`
4. Use larger `device_train_batch_size` if memory allows
5. Consider using a smaller model for prototyping

### Poor Quality Results

1. Increase `rank` and `lora_alpha`
2. Add validation dataset
3. Increase `epochs`
4. Lower `learning_rate`
5. Check data quality and format
