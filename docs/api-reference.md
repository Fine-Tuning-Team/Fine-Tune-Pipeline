# API Reference

This page provides detailed API documentation for the Fine-Tune Pipeline components.

## Core Classes

### FineTune

The main class for fine-tuning language models.

::: app.finetuner.FineTune
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### ConfigManager

Centralized configuration management.

::: app.config_manager.ConfigManager
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Configuration Classes

### FineTunerConfig

Configuration dataclass for fine-tuning parameters.

::: app.config_manager.FineTunerConfig
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Utility Functions

### load_huggingface_dataset

Load datasets from Hugging Face Hub.

::: app.utils.load_huggingface_dataset
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### login_huggingface

Authenticate with Hugging Face Hub.

::: app.utils.login_huggingface
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### setup_run_name

Generate unique run names for experiments.

::: app.utils.setup_run_name
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Usage Examples

### Basic Fine-Tuning

```python
from app.finetuner import FineTune
from app.config_manager import get_config_manager

# Initialize with default configuration
tuner = FineTune()

# Run fine-tuning
stats = tuner.run()
print(f"Training completed with stats: {stats}")
```

### Custom Configuration

```python
from app.config_manager import ConfigManager, FineTunerConfig
from app.finetuner import FineTune

# Load custom configuration
config_manager = ConfigManager("custom_config.toml")
config = FineTunerConfig.from_config(config_manager)

# Initialize tuner with custom config
tuner = FineTune(config_manager=config_manager)

# Access configuration
print(f"Base model: {config.base_model_id}")
print(f"Epochs: {config.epochs}")
print(f"Learning rate: {config.learning_rate}")

# Run training
stats = tuner.run()
```

### Programmatic Configuration

```python
from app.config_manager import FineTunerConfig
from app.finetuner import FineTune

# Create configuration programmatically
config = FineTunerConfig(
    base_model_id="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
    training_data_id="your-username/your-dataset",
    epochs=3,
    learning_rate=0.0002,
    device_train_batch_size=4,
    rank=16,
    lora_alpha=16
)

# Use configuration
tuner = FineTune(config=config)
stats = tuner.run()
```

## Error Handling

The API includes comprehensive error handling:

```python
from app.finetuner import FineTune
from app.config_manager import ConfigManager

try:
    config_manager = ConfigManager("config.toml")
    tuner = FineTune(config_manager=config_manager)
    stats = tuner.run()
    
except FileNotFoundError as e:
    print(f"Configuration file not found: {e}")
    
except ValueError as e:
    print(f"Invalid configuration: {e}")
    
except RuntimeError as e:
    print(f"Training error: {e}")
    
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Type Hints

The API uses comprehensive type hints for better IDE support:

```python
from typing import Dict, Any, Optional, Tuple
from transformers import PreTrainedTokenizerBase
from unsloth import FastLanguageModel

def load_model_and_tokenizer(
    model_id: str,
    max_seq_length: int = 4096,
    dtype: Optional[str] = None
) -> Tuple[FastLanguageModel, PreTrainedTokenizerBase]:
    """Load model and tokenizer with type safety."""
    pass

def process_dataset(
    dataset_id: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Process dataset with configuration."""
    pass
```

## Constants and Enums

### Model Types

```python
SUPPORTED_MODEL_TYPES = [
    "qwen",
    "llama", 
    "mistral",
    "gemma",
    "phi"
]
```

### Optimizers

```python
SUPPORTED_OPTIMIZERS = [
    "adamw_torch",
    "adamw_hf",
    "paged_adamw_8bit",
    "paged_adamw_32bit"
]
```

### Schedulers

```python
SUPPORTED_SCHEDULERS = [
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup"
]
```

## Configuration Schema

### Complete Configuration Example

```python
{
    "fine_tuner": {
        # Model configuration
        "base_model_id": str,
        "max_sequence_length": int,
        "dtype": Optional[str],
        "load_in_4bit": bool,
        "load_in_8bit": bool,
        "full_finetuning": bool,
        
        # LoRA configuration
        "rank": int,
        "lora_alpha": int,
        "lora_dropout": float,
        "target_modules": List[str],
        "bias": str,
        "use_rslora": bool,
        "loftq_config": Optional[str],
        
        # Dataset configuration
        "training_data_id": str,
        "validation_data_id": Optional[str],
        "dataset_num_proc": int,
        "question_column": str,
        "ground_truth_column": str,
        "system_prompt_column": Optional[str],
        "system_prompt_override_text": Optional[str],
        
        # Training configuration
        "epochs": int,
        "learning_rate": float,
        "device_train_batch_size": int,
        "device_validation_batch_size": int,
        "grad_accumulation": int,
        "warmup_steps": int,
        "optimizer": str,
        "weight_decay": float,
        "lr_scheduler_type": str,
        "seed": int,
        
        # Logging configuration
        "log_steps": int,
        "log_first_step": bool,
        "save_steps": int,
        "save_total_limit": int,
        "push_to_hub": bool,
        "report_to": str,
        "wandb_project_name": str,
        
        # Advanced configuration
        "packing": bool,
        "use_gradient_checkpointing": Union[bool, str],
        "use_flash_attention": bool,
        "train_on_responses_only": bool,
        "question_part": str,
        "answer_part": str,
        
        # Run naming
        "run_name": Optional[str],
        "run_name_prefix": str,
        "run_name_suffix": str
    }
}
```

## Environment Variables

The API recognizes these environment variables:

| Variable | Purpose | Required |
|----------|---------|----------|
| `HF_TOKEN` | Hugging Face authentication | Yes |
| `WANDB_TOKEN` | Weights & Biases API key | Yes |
| `OPENAI_API_KEY` | OpenAI API key for evaluation | Optional |
| `TRANSFORMERS_CACHE` | HuggingFace cache directory | No |
| `CUDA_VISIBLE_DEVICES` | GPU device selection | No |

## Performance Tips

### Memory Optimization

```python
# For low-memory systems
config = {
    "load_in_4bit": True,
    "device_train_batch_size": 1,
    "grad_accumulation": 16,
    "use_gradient_checkpointing": "unsloth",
    "max_sequence_length": 1024
}
```

### Speed Optimization

```python
# For faster training
config = {
    "packing": True,
    "use_flash_attention": True,
    "dataset_num_proc": 8,
    "dtype": None  # Auto-select best precision
}
```

### Quality Optimization

```python
# For better results
config = {
    "rank": 64,
    "lora_alpha": 32,
    "epochs": 5,
    "learning_rate": 0.0001,
    "validation_data_id": "your-validation-set"
}
```
