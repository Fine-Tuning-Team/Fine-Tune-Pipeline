# Quick Start Guide

This guide will walk you through running your first fine-tuning job with the Fine-Tune Pipeline in just a few minutes.

## Prerequisites

Before you begin, make sure you have:

- ✅ [Set up your environment](environment-setup.md) with API keys
- ✅ [Configured your MLFlow server](choreo-setup.md) to log experiments

## Step 1: Navigate to the Github repository and branch

First, go to the GitHub repository of the pipeline and switch to the branch which aligns with the model you are trying to fine-tune. For example, if you are working with the Qwen2.5 model, switch to the `lora-qwen2.5` branch.

If such a branch does not exist, make a branch from the `lora-dev` branch and name it according to the model you are working with, e.g., `lora-model_XYZ`.

## Step 2: Understanding the Default Configuration

In the files, you will find the `config.toml` file. The pipeline comes with a pre-configured setup in `config.toml`. Let's look at the key settings:

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
run_name = "0.0.1"  # Increment this for each run
```

!!! tip "First Run Recommendation"
    The default configuration is designed for a quick first run. It uses a small model and dataset that should complete training in 10-15 minutes on a modern GPU.

## Step 3: Run Your First Fine-Tuning Job

Make a small change to the `config.toml` file. For example, bump the `run_name` under `[MLFLOW]` section by 0.0.1.

This will trigger the pipeline to run. This will consist of 3 stages: `fine-tuning`, `inference`, and `evaluation`.

### 1. Fine-tuning

#### 1.1 What Happens During Fine Tuning

1. **Model Loading**: Downloads and loads the base model (Qwen2.5-0.5B)
2. **Data Processing**: Downloads and processes the training dataset
3. **LoRA Setup**: Configures Low-Rank Adaptation for efficient fine-tuning
4. **Training**: Runs 3 epochs of training with progress tracking
5. **Saving**: Saves the model locally and pushes to Hugging Face Hub

#### 1.2 Expected Output

You should see a final output similar to this in github actions:

```text
--- ✅ Fine-tuning completed successfully. ---
```

### 2. Inference

#### 2.1 What Happens During Inferencing

After training, the pipeline will automatically run inference. This involves:

1. **Model Loading**: Loads the fine-tuned model
2. **Data Preparation**: Downloads and processes the test dataset for inference
3. **Inference Execution**: Runs inference with the configured parameters in `config.toml`
4. **Output Generation**: Saves results in JSONL format
5. **Pushing Results**: Uploads inference results to Hugging Face Hub

#### 2.2 Expected Output

You should see a final output similar to this in github actions:

```text
--- ✅ Inference completed successfully. ---
```

### 3. Evaluation

#### 3.1 What Happens During Evaluation

After inference, the pipeline will automatically run evaluation. This includes:

1. **Loading Results**: Loads the inference output
2. **Evaluation Metrics**: Computes various metrics like Factual Correctness, Answer Accuracy, and more with RAGAS
3. **Reporting**: Generates detailed reports in Excel and JSON formats
4. **Logging**: Saves evaluation metrics to MLflow
5. **Pushing Results**: Uploads evaluation results to Hugging Face Hub

#### 3.2 Expected Output

You should see a final output similar to this in github actions:

```text
--- ✅ Evaluation completed successfully. ---
```

## Next Steps

Congratulations! 🎉 You've successfully run your first fine-tuning pipeline. Here's what you can do next:

### Customize Your Training

1. **Use Your Own Data**: Replace `training_data_id`, `testing_data_id` with your datasets
2. **Try Different Models**: Experiment with larger models like Llama, Gemma by changing `base_model_id`
3. **Adjust Hyperparameters**: Modify learning rate, batch size, epochs etc.
4. **Explore Advanced Features**: Check out the [Advanced Configuration](../tutorials/advanced-configuration.md) guide

### See Also

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
    For faster training:
    ```toml
    learning_rate = 0.0005  # Increase learning rate
    epochs = 2              # Reduce number of epochs
    device_train_batch_size = 8  # Increase batch size if GPU allows
    ```

Happy fine-tuning! 🚀
