# Troubleshooting

This guide helps you diagnose and fix common issues with the Fine-Tune Pipeline.

## Installation Issues

### uv sync fails

**Error**: `Failed to resolve dependencies`

**Solutions**:
```bash
# Update uv to latest version
uv self update

# Clear cache and retry
uv cache clean
uv sync --refresh

# Check Python version
python --version  # Should be 3.12+
```

### ImportError: No module named 'unsloth'

**Error**: `ModuleNotFoundError: No module named 'unsloth'`

**Solutions**:
```bash
# Reinstall unsloth with CUDA support
uv remove unsloth
uv add "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"

# Or for CUDA 12.1
uv add "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
```

## Configuration Issues

### FileNotFoundError: config.toml not found

**Error**: `Configuration file not found: config.toml`

**Solutions**:
- Ensure `config.toml` is in the project root
- Check file permissions
- Verify you're in the correct directory

### Invalid configuration values

**Error**: `ValueError: Invalid configuration parameter`

**Solutions**:
- Check TOML syntax (no missing quotes, brackets)
- Verify data types match expected format
- Use `"null"` for optional parameters

## Memory Issues

### CUDA out of memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:

#### Immediate Fix
```toml
[fine_tuner]
device_train_batch_size = 1
grad_accumulation = 16
max_sequence_length = 1024
use_gradient_checkpointing = "unsloth"
```

#### Progressive Optimization
1. **Reduce batch size**:
   ```toml
   device_train_batch_size = 1  # From 4 to 1
   ```

2. **Enable gradient checkpointing**:
   ```toml
   use_gradient_checkpointing = "unsloth"
   ```

3. **Reduce sequence length**:
   ```toml
   max_sequence_length = 1024  # From 4096
   ```

4. **Use higher quantization**:
   ```toml
   load_in_4bit = true
   load_in_8bit = false
   ```

#### Memory Requirements by GPU

| GPU | VRAM | Recommended Settings |
|-----|------|---------------------|
| RTX 3060 | 12GB | batch_size=1, seq_len=1024, 4-bit |
| RTX 3070 | 8GB | batch_size=2, seq_len=2048, 4-bit |
| RTX 3080 | 10GB | batch_size=4, seq_len=4096, 4-bit |
| RTX 4090 | 24GB | batch_size=8, seq_len=4096, 4-bit |

### CPU Training (No GPU)

```toml
[fine_tuner]
device_train_batch_size = 1
max_sequence_length = 512
epochs = 1
dtype = "float32"
load_in_4bit = false
load_in_8bit = false
```

## Authentication Issues

### Hugging Face authentication failed

**Error**: `HfHubError: 401 Client Error`

**Solutions**:
```bash
# Login with CLI
huggingface-cli login

# Or set environment variable
export HF_TOKEN="your_token_here"

# Verify authentication
python -c "from huggingface_hub import whoami; print(whoami())"
```

### Weights & Biases login failed

**Error**: `wandb: ERROR Unable to authenticate`

**Solutions**:
```bash
# Login interactively
wandb login

# Or set environment variable
export WANDB_TOKEN="your_api_key"

# Verify authentication
python -c "import wandb; wandb.login()"
```

## Dataset Issues

### Dataset not found

**Error**: `DatasetNotFoundError: Dataset 'username/dataset' not found`

**Solutions**:
- Verify dataset name and username are correct
- Check if dataset is public or you have access
- Ensure proper authentication for private datasets

### Dataset format errors

**Error**: `KeyError: 'question'` or `KeyError: 'answer'`

**Solutions**:
- Check column names in your dataset
- Update configuration to match your dataset:
  ```toml
  question_column = "your_question_column"
  ground_truth_column = "your_answer_column"
  ```

### Dataset loading too slow

**Solutions**:
```toml
[fine_tuner]
dataset_num_proc = 8  # Increase parallel processing
```

## Training Issues

### Training stuck or very slow

**Symptoms**: Training doesn't progress or is extremely slow

**Solutions**:
1. **Enable optimizations**:
   ```toml
   packing = true
   use_flash_attention = true
   ```

2. **Check data loading**:
   ```toml
   dataset_num_proc = 4  # Adjust based on CPU cores
   ```

3. **Verify GPU usage**:
   ```bash
   nvidia-smi  # Check GPU utilization
   ```

### Loss not decreasing

**Symptoms**: Training loss remains constant or increases

**Solutions**:
1. **Adjust learning rate**:
   ```toml
   learning_rate = 0.0001  # Reduce from 0.0002
   ```

2. **Check data quality**:
   - Verify dataset format
   - Check for duplicate or malformed entries

3. **Increase model capacity**:
   ```toml
   rank = 32  # Increase from 16
   lora_alpha = 32
   ```

### Model outputs gibberish

**Symptoms**: Generated text is incoherent

**Solutions**:
1. **Check chat template**:
   - Verify model supports chat format
   - Check system prompt configuration

2. **Adjust generation parameters**:
   ```toml
   [inferencer]
   temperature = 0.7  # Reduce from 1.0
   max_new_tokens = 150  # Limit output length
   ```

## Inference Issues

### Model not found for inference

**Error**: `Model not found: username/model-name`

**Solutions**:
- Check if model was uploaded to Hub
- Verify model name format: `{hf_user_id}/{run_name}`
- Ensure model upload completed successfully

### Inference too slow

**Solutions**:
```toml
[inferencer]
load_in_4bit = true
use_cache = true
device_batch_size = 1
```

## Evaluation Issues

### OpenAI API errors

**Error**: `openai.RateLimitError` or `openai.AuthenticationError`

**Solutions**:
- Check API key is valid and has credits
- Reduce evaluation batch size
- Use alternative metrics (BLEU, ROUGE) instead

### Evaluation metrics seem wrong

**Solutions**:
- Verify ground truth format matches predictions
- Check evaluation dataset column mapping
- Review metric definitions and expected ranges

## Performance Optimization

### Speed up training

1. **Enable optimizations**:
   ```toml
   packing = true
   use_flash_attention = true
   use_gradient_checkpointing = "unsloth"
   ```

2. **Optimize data loading**:
   ```toml
   dataset_num_proc = 8
   ```

3. **Use appropriate precision**:
   ```toml
   dtype = "null"  # Auto-select best precision
   ```

### Improve model quality

1. **Increase model capacity**:
   ```toml
   rank = 64
   lora_alpha = 32
   ```

2. **More training**:
   ```toml
   epochs = 5
   learning_rate = 0.0001
   ```

3. **Better data**:
   - Use validation set
   - Increase dataset size
   - Improve data quality

## Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export TRANSFORMERS_VERBOSITY=debug
export DATASETS_VERBOSITY=debug
```

## Getting Help

If you're still experiencing issues:

1. **Check the error message carefully** - it often contains specific guidance
2. **Search existing issues** on GitHub
3. **Check component logs** in Weights & Biases
4. **Create a minimal reproduction case**
5. **Open an issue** with:
   - Full error message
   - Configuration file
   - Environment details (GPU, Python version)
   - Steps to reproduce

## Common Error Patterns

### Pattern: "torch.cuda.OutOfMemoryError"
**Cause**: GPU memory exhaustion  
**Fix**: Reduce batch size or sequence length

### Pattern: "ModuleNotFoundError"
**Cause**: Missing dependencies  
**Fix**: Run `uv sync` or install specific package

### Pattern: "HfHubError: 401"
**Cause**: Authentication failure  
**Fix**: Set proper API tokens

### Pattern: "ValueError: Invalid configuration"
**Cause**: TOML syntax or type errors  
**Fix**: Check configuration format

### Pattern: "DatasetNotFoundError"
**Cause**: Dataset access or naming issues  
**Fix**: Verify dataset exists and permissions
