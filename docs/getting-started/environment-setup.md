# Environment Setup

Before running the Fine-Tune Pipeline, you need to set up API keys and environment variables for various services.

## Required API Keys

### 1. Hugging Face Token  

Required for accessing models and datasets from Hugging Face Hub.

1. **Create a Hugging Face account** at [huggingface.co](https://huggingface.co)
2. **Generate a token**:
   - Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - Click "New token"
   - Choose "Write" permissions for model uploads
   - Copy the generated token

### 2. Weights & Biases API Key

Required for experiment tracking and logging.

1. **Create a W&B account** at [wandb.ai](https://wandb.ai)
2. **Get your API key**:
   - Go to [Settings](https://wandb.ai/settings)
   - Copy your API key from the "API keys" section

### 3. OpenAI API Key (For Evaluation)

Required for LLM-based evaluation metrics.

1. **Create an OpenAI account** at [platform.openai.com](https://platform.openai.com)
2. **Generate an API key**:
   - Go to [API Keys](https://platform.openai.com/api-keys)
   - Click "Create new secret key"
   - Copy the generated key

## Setting Up Environment Variables

### Method 1: Command Line Arguments (Recommended)

You can pass API keys directly when running the pipeline:

```bash
# For fine-tuning
uv run app/finetuner.py --hf-key "your_hf_token" --wandb-key "your_wandb_key"

# For inference  
uv run app/inferencer.py --hf-key "your_hf_token"

# For evaluation
uv run app/evaluator.py --openai-key "your_openai_key"
```

### Method 2: Environment Variables

Set environment variables in your shell:

#### Windows (Command Prompt)

```cmd
set HF_TOKEN=your_hf_token_here
set WANDB_TOKEN=your_wandb_key_here
set OPENAI_API_KEY=your_openai_key_here
```

#### Windows (PowerShell)

```powershell
$env:HF_TOKEN="your_hf_token_here"
$env:WANDB_TOKEN="your_wandb_key_here" 
$env:OPENAI_API_KEY="your_openai_key_here"
```

#### macOS/Linux (Bash/Zsh)

```bash
export HF_TOKEN="your_hf_token_here"
export WANDB_TOKEN="your_wandb_key_here"
export OPENAI_API_KEY="your_openai_key_here"
```

### Method 3: Environment File

Create a `.env` file in the project root (make sure to add it to `.gitignore`):

```bash
# .env file
HF_TOKEN=your_hf_token_here
WANDB_TOKEN=your_wandb_key_here
OPENAI_API_KEY=your_openai_key_here
```

## Configuration File Setup

The pipeline uses `config.toml` for configuration. You can customize it for your needs:

### Basic Configuration

```toml
[fine_tuner]
# Model settings
base_model_id = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
max_sequence_length = 4096

# Training data
training_data_id = "your-username/your-training-dataset"
validation_data_id = "your-username/your-validation-dataset"  # Optional

# Training parameters
epochs = 3
learning_rate = 0.0002
device_train_batch_size = 4

# Weights & Biases
wandb_project_name = "your-project-name"
```

### Verify Setup

Test your environment setup:

```bash
# Test Hugging Face authentication
uv run python -c "from huggingface_hub import whoami; print(f'Logged in as: {whoami()}')"

# Test Weights & Biases
uv run python -c "import wandb; wandb.login(); print('W&B authentication successful')"

# Test the pipeline configuration
uv run python -c "from app.config_manager import get_config_manager; cm = get_config_manager(); print('Config loaded successfully')"
```

## GPU Configuration

If you're using a GPU for training, verify your setup:

```bash
# Check CUDA availability
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Check unsloth GPU support
uv run python -c "from unsloth import is_bfloat16_supported; print(f'bfloat16 supported: {is_bfloat16_supported()}')"
```

## Next Steps

With your environment set up, you're ready to:

1. [Run your first fine-tuning job](quick-start.md)
2. [Explore configuration options](../configuration/overview.md)
3. [Learn about advanced features](../tutorials/advanced-configuration.md)

## Security Best Practices

!!! warning "API Key Security"
    - Never commit API keys to version control
    - Use environment variables or command-line arguments
    - Add `.env` files to your `.gitignore`
    - Rotate API keys regularly
    - Use minimal required permissions for each key

!!! tip "Production Deployment"
    For production deployments, consider using:
    - Secret management services (AWS Secrets Manager, Azure Key Vault)
    - CI/CD environment variables
    - Kubernetes secrets
    - Docker secrets
