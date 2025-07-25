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

### 2. Weights & Biases API Key (No need at the moment; you can skip this step)

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

### 4. Runpod API Key (For GPU Instances)

Required for running fine-tuning jobs on Runpod GPU instances.

1. **Create a Runpod account** at [runpod.io](https://runpod.io)
2. **Generate an API key**:
   - Go to runpod account [runpod.io](https://console.runpod.io/)
   - Go to `Settings > API Keys`
   - Generate and copy the generated key

### 5. SSH Keys (For Remote Access)

Required for accessing remote servers or instances. Generate the SSH keys (public and private) and copy them.

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

## Setting Up Environment Variables

In GitHub, go to `Settings > Secrets and variables > Actions` and add the following

```toml
HF_TOKEN=your_hf_token_here
WANDB_TOKEN=your_wandb_key_here (You can skip this secret for now)
OPENAI_API_KEY=your_openai_key_here
RUNPOD_API_KEY=your_runpod_api_key_here
SSH_PRIVATE_KEY=your_ssh_private_key_here
SSH_PUBLIC_KEY=your_ssh_public_key_here
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
training_data_id = "your-huggingface-username/your-training-dataset"
validation_data_id = "your-huggingface-username/your-validation-dataset"  # Optional

# Training parameters
epochs = 3
learning_rate = 0.0002
device_train_batch_size = 4

[inferencer]
# Model settings
max_sequence_length = 4096
max_new_tokens = 512
temperature = 0.7
min_p = 0.1

# Hugging Face user ID
hf_user_id = "your-huggingface-username"

[evaluator]
# Metrics settings
metrics = ["bleu_score", "rouge_score", "factual_correctness"]

# Hugging Face user ID
hf_user_id = "your-huggingface-username"

[mlflow]
# MLflow settings
tracking_uri = "https://your-mlflow-tracking-uri"
experiment_name = "your-experiment-name"
run_name = "your-run-name"
```

## Next Steps

With your environment set up, you're ready to:

1. [Run your first fine-tuning job](quick-start.md)
2. [Explore configuration options](../configuration/overview.md)
3. [Learn about advanced features](../tutorials/advanced-configuration.md)
4. [Set up MLFlow server](choreo-setup.md)
