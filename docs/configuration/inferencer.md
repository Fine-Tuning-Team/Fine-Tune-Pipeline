# Inferencer Configuration

The `[inferencer]` section controls the model inference and prediction process. This component loads your fine-tuned model and generates predictions on test datasets.

## Basic Configuration

```toml
[inferencer]
# Model settings
max_sequence_length = 4096
dtype = "null"  # Options: "float16", "bfloat16", null
load_in_4bit = true
load_in_8bit = false

# Dataset configuration
testing_data_id = "your-username/test-dataset"
question_column = "question"
ground_truth_column = "answer"
system_prompt_column = "null"  # Optional
system_prompt_override_text = "null"  # Optional

# Generation parameters
max_new_tokens = 512
use_cache = true
temperature = 0.7
min_p = 0.1

# Model location
hf_user_id = "your-username"
run_name = "null"  # Use latest model if null
run_name_prefix = ""
run_name_suffix = ""
```

## Model Configuration

### Model Loading Settings

```toml
[inferencer]
# Maximum sequence length for inference
max_sequence_length = 4096  # Should match training settings

# Data type for model weights
dtype = "null"  # Auto-detect optimal precision

# Quantization (choose one)
load_in_4bit = true   # Most memory efficient
load_in_8bit = false  # Balanced option
```

### Model Location

The inferencer automatically constructs the model path using:

```
{hf_user_id}/{run_name}
```

If `run_name` is "null", it uses the most recent model from your account.

## Dataset Configuration

### Data Source

```toml
[inferencer]
# Test dataset from Hugging Face Hub
testing_data_id = "your-username/test-dataset"

# Column mapping
question_column = "question"      # Input column
ground_truth_column = "answer"    # Expected output (for comparison)
```

### System Prompts

Control how the model receives instructions:

```toml
[inferencer]
# Use column from dataset
system_prompt_column = "system_prompt"

# Or override with custom prompt
system_prompt_override_text = "You are a helpful assistant specialized in..."
```

## Generation Parameters

### Basic Generation

```toml
[inferencer]
# Maximum tokens to generate
max_new_tokens = 512

# Enable key-value caching for speed
use_cache = true
```

### Advanced Generation

```toml
[inferencer]
# Temperature (0.0 = deterministic, 1.0 = creative)
temperature = 0.7

# Min-p sampling (alternative to top-p)
min_p = 0.1

# Additional parameters (if needed)
top_p = 0.9
top_k = 50
repetition_penalty = 1.1
```

## Performance Tuning

### Memory Optimization

```toml
[inferencer]
# For low-memory systems
load_in_4bit = true
max_sequence_length = 2048
max_new_tokens = 256
```

### Speed Optimization

```toml
[inferencer]
# Enable caching
use_cache = true

# Optimal sequence length
max_sequence_length = 4096

# Batch processing (if supported)
batch_size = 1
```

## Output Configuration

The inferencer outputs results to `inferencer_output.jsonl` with this format:

```json
{
    "question": "What is machine learning?",
    "predicted_answer": "Machine learning is a subset of artificial intelligence...",
    "ground_truth": "ML is a method of data analysis...",
    "metadata": {
        "model_id": "your-username/model-name",
        "generation_config": {...},
        "inference_time": 1.23
    }
}
```

## Usage Examples

### Basic Usage

```bash
# Run inference with default settings
uv run app/inferencer.py --hf-key "your_token"
```

### Custom Configuration

```bash
# Use specific model
export HF_TOKEN="your_token"
uv run app/inferencer.py
```

### Programmatic Usage

```python
from app.inferencer import Inferencer

# Initialize with configuration
inferencer = Inferencer()

# Run inference
results = inferencer.run()
print(f"Generated {len(results)} predictions")
```

## Common Configurations

### Research/Development

```toml
[inferencer]
max_new_tokens = 150
temperature = 0.1  # More deterministic
load_in_4bit = true
```

### Production Inference

```toml
[inferencer]
max_new_tokens = 512
temperature = 0.7
use_cache = true
max_sequence_length = 4096
```

### Batch Processing

```toml
[inferencer]
# Optimize for throughput
use_cache = false
max_new_tokens = 256
temperature = 0.0  # Deterministic
```

## Error Handling

Common issues and solutions:

### Model Not Found
```
Error: Model not found: username/model-name
```
**Solution**: Check model was uploaded and name is correct

### Out of Memory
```
Error: CUDA out of memory
```
**Solution**: Enable quantization and reduce sequence length
```toml
load_in_4bit = true
max_sequence_length = 2048
```

### Generation Issues
```
Error: Generated text is empty
```
**Solution**: Adjust generation parameters
```toml
max_new_tokens = 100  # Increase
temperature = 0.8     # Add randomness
```
