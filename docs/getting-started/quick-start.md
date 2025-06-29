# Quick Start Guide

This guide will walk you through running your first fine-tuning job with the Fine-Tune Pipeline in just a few minutes.

## Prerequisites

Before you begin, make sure you have:

- ✅ [Installed the pipeline](installation.md)
- ✅ [Set up your environment](environment-setup.md) with API keys
- ✅ A GPU (recommended) or CPU for training

## Step 1: Verify Your Setup

First, let's make sure everything is working:

```bash
# Navigate to your project directory
cd Fine-Tune-Pipeline

# Sync dependencies
uv sync

# Test the installation
uv run python -c "from app.finetuner import FineTune; print('✅ Setup verified!')"
```

## Step 2: Understanding the Default Configuration

The pipeline comes with a pre-configured setup in `config.toml`. Let's look at the key settings:

```toml
[fine_tuner]
# Base model - A small, efficient model for quick testing
base_model_id = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"

# Training data - Default dataset for question-answering
training_data_id = "rtweera/simple_implicit_n_qa_results_v2"

# Training settings - Optimized for quick runs
epochs = 3
device_train_batch_size = 4
learning_rate = 0.0002
```

!!! tip "First Run Recommendation"
    The default configuration is designed for a quick first run. It uses a small model and dataset that should complete training in 10-15 minutes on a modern GPU.

## Step 3: Run Your First Fine-Tuning Job

Now let's run the fine-tuner with your API keys:

```bash
uv run app/finetuner.py --hf-key "your_hf_token" --wandb-key "your_wandb_key"
```

### What Happens During Training

1. **Model Loading**: Downloads and loads the base model (Qwen2.5-0.5B)
2. **Data Processing**: Downloads and processes the training dataset
3. **LoRA Setup**: Configures Low-Rank Adaptation for efficient fine-tuning
4. **Training**: Runs 3 epochs of training with progress tracking
5. **Saving**: Saves the model locally and pushes to Hugging Face Hub

### Expected Output

You should see output similar to this:

```text
--- ✅ Login to Hugging Face Hub successful. ---
--- ✅ Training dataset loaded: rtweera/simple_implicit_n_qa_results_v2 ---
--- ✅ No validation dataset provided. Skipping validation. ---
--- ✅ Model and tokenizer loaded successfully. ---
--- ✅ Data preprocessing completed. ---
Run name set to: fine-tuned-model-20250629-143022
--- ✅ Weights & Biases setup completed. ---
--- ✅ Trainer initialized successfully. ---
--- ✅ Starting training... ---

Training Progress:
  0%|          | 0/150 [00:00<?, ?it/s]
 10%|█         | 15/150 [00:30<04:30,  2.0s/it]
 20%|██        | 30/150 [01:00<04:00,  2.0s/it]
...

--- ✅ Training completed with stats: {...} ---
--- ✅ Model and tokenizer saved to ./models/fine_tuned locally and to Hugging Face Hub ---
--- ✅ Fine-tuning completed successfully. ---
```

## Step 4: Test Your Fine-Tuned Model

After training, let's test the model with inference:

### 4.1 Update Configuration for Inference

First, update your `config.toml` to use your newly trained model:

```toml
[inferencer]
# Use your Hugging Face username and the generated model name
hf_user_id = "your-hf-username"
# The run_name from training (or leave as "null" to use the latest)
run_name = "null"

# Test dataset
testing_data_id = "rtweera/user_centric_results_v2"
```

### 4.2 Run Inference

```bash
uv run app/inferencer.py --hf-key "your_hf_token"
```

This will generate predictions and save them to `inferencer_output.jsonl`.

## Step 5: Evaluate Your Model

Finally, let's evaluate how well your model performed:

```bash
uv run app/evaluator.py --openai-key "your_openai_key"
```

This will generate:

- `evaluator_output_summary.json` - Overall performance metrics
- `evaluator_output_detailed.xlsx` - Detailed evaluation results

## Step 6: Review Results

### Weights & Biases Dashboard

1. Go to [wandb.ai](https://wandb.ai)
2. Navigate to your project: `fine-tuning-project-ci-cd`
3. View training metrics, loss curves, and system metrics

### Local Results

Check the generated files:

```bash
# View inference results
head -5 inferencer_output.jsonl

# View evaluation summary
cat evaluator_output_summary.json

# Open detailed results in Excel
start evaluator_output_detailed.xlsx  # Windows
# or
open evaluator_output_detailed.xlsx   # macOS
```

## Next Steps

Congratulations! 🎉 You've successfully run your first fine-tuning pipeline. Here's what you can do next:

### Customize Your Training

1. **Use Your Own Data**: Replace `training_data_id` with your dataset
2. **Try Different Models**: Experiment with larger models like Llama or Mistral
3. **Adjust Hyperparameters**: Modify learning rate, batch size, epochs

### Advanced Features

1. **[Advanced Configuration](../tutorials/advanced-configuration.md)** - Explore all configuration options
2. **[CI/CD Integration](../tutorials/ci-cd-integration.md)** - Set up automated training pipelines
3. **[API Reference](../api-reference.md)** - Deep dive into the codebase

### Troubleshooting

If you encounter issues:

1. Check the [Troubleshooting Guide](../troubleshooting.md)
2. Verify your API keys are correct
3. Ensure you have sufficient GPU memory
4. Check the console output for specific error messages

## Common First-Run Issues

!!! warning "Out of Memory"
    If you get CUDA out of memory errors, reduce the batch size:
    ```toml
    device_train_batch_size = 2  # Reduce from 4
    grad_accumulation = 8        # Increase to maintain effective batch size
    ```

!!! warning "Dataset Not Found"
    If the dataset fails to load, check:
    - Your internet connection
    - The dataset ID is correct
    - You have access to the dataset (some require authentication)

!!! tip "Training Too Slow"
    For faster training on CPU:
    ```toml
    device_train_batch_size = 1
    epochs = 1
    max_sequence_length = 1024  # Reduce sequence length
    ```

Happy fine-tuning! 🚀
