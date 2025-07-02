# Basic Fine-Tuning Tutorial

This tutorial will guide you through your first fine-tuning project using the Fine-Tune Pipeline. We'll fine-tune a small language model on a question-answering dataset.

## Prerequisites

Before starting this tutorial, ensure you have:

- ✅ [Installed the pipeline](../getting-started/installation.md)
- ✅ [Set up your environment](../getting-started/environment-setup.md)
- ✅ API keys for Hugging Face and Weights & Biases
- ✅ A GPU with at least 4GB VRAM (or CPU for slower training)

## Tutorial Overview

In this tutorial, we will:

1. Prepare a custom dataset
2. Configure the pipeline
3. Run fine-tuning
4. Test the fine-tuned model
5. Evaluate results

**Estimated Time**: 30-45 minutes

## Step 1: Prepare Your Dataset

We'll create a simple question-answering dataset about programming concepts.

### Create a Dataset File

Create a file called `programming_qa.jsonl`:

```json
{"question": "What is a variable in programming?", "answer": "A variable is a named storage location in memory that holds a value which can be modified during program execution."}
{"question": "What is a function?", "answer": "A function is a reusable block of code that performs a specific task and can accept inputs (parameters) and return outputs."}
{"question": "What is object-oriented programming?", "answer": "Object-oriented programming (OOP) is a programming paradigm based on the concept of objects, which contain data (attributes) and code (methods)."}
{"question": "What is an algorithm?", "answer": "An algorithm is a step-by-step procedure or set of instructions designed to solve a specific problem or perform a particular task."}
{"question": "What is debugging?", "answer": "Debugging is the process of finding, analyzing, and fixing errors or bugs in computer programs to ensure they work correctly."}
```

### Upload to Hugging Face Hub

1. **Create a new dataset repository** on [Hugging Face Hub](https://huggingface.co/new-dataset)
2. **Upload your dataset**:

```bash
# Install Hugging Face CLI if not already installed
pip install huggingface_hub[cli]

# Login to Hugging Face
huggingface-cli login

# Create and upload dataset
huggingface-cli repo create your-username/programming-qa-dataset --type dataset
huggingface-cli upload your-username/programming-qa-dataset programming_qa.jsonl
```

Alternatively, you can use the web interface to upload your file.

## Step 2: Configure the Pipeline

Edit your `config.toml` file with the following configuration:

```toml
[fine_tuner]
# Use a small, efficient model for this tutorial
base_model_id = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
max_sequence_length = 2048

# Memory-efficient settings
load_in_4bit = true
load_in_8bit = false
full_finetuning = false

# LoRA configuration - conservative settings for first run
rank = 16
lora_alpha = 16
lora_dropout = 0.1
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
bias = "none"

# Your dataset
training_data_id = "your-username/programming-qa-dataset"
validation_data_id = "null"  # No validation set for this tutorial

# Dataset configuration
dataset_num_proc = 2
question_column = "question"
ground_truth_column = "answer"
system_prompt_column = "null"
system_prompt_override_text = "You are a helpful programming tutor. Provide clear and concise explanations."

# Training parameters - short training for tutorial
epochs = 2
learning_rate = 0.0003
device_train_batch_size = 2
device_validation_batch_size = 2
grad_accumulation = 4

# Optimization settings
warmup_steps = 10
optimizer = "paged_adamw_8bit"
weight_decay = 0.01
lr_scheduler_type = "linear"
seed = 42

# Logging and saving
log_steps = 5
log_first_step = true
save_steps = 50
save_total_limit = 2
push_to_hub = true

# Weights & Biases
wandb_project_name = "programming-qa-tutorial"
report_to = "wandb"

# Run naming
run_name = "null"
run_name_prefix = "programming-qa-"
run_name_suffix = ""

# Advanced settings
packing = false
use_gradient_checkpointing = "unsloth"
use_flash_attention = true
use_rslora = false
loftq_config = "null"

# Response-only training for chat models
train_on_responses_only = true
question_part = "<|im_start|>user\n"
answer_part = "<|im_start|>assistant\n"
```

### Configuration Explanation

| Parameter | Value | Explanation |
|----------|-------|-------------|
| `base_model_id` | `unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit` | Small model for quick training |
| `epochs` | `2` | Short training for tutorial |
| `rank` | `16` | Moderate LoRA rank for good quality |
| `device_train_batch_size` | `2` | Small batch size for memory efficiency |
| `system_prompt_override_text` | Custom prompt | Specialized for programming help |

## Step 3: Run Fine-Tuning

Now let's start the fine-tuning process:

```bash
# Navigate to your project directory
cd Fine-Tune-Pipeline

# Ensure dependencies are synced
uv sync

# Run fine-tuning with your API keys
uv run app/finetuner.py --hf-key "your_hf_token" --wandb-key "your_wandb_key"
```

### What to Expect

The training should take approximately 10-15 minutes on a modern GPU. You'll see output like:

```text
--- ✅ Login to Hugging Face Hub successful. ---
--- ✅ Training dataset loaded: your-username/programming-qa-dataset ---
--- ✅ No validation dataset provided. Skipping validation. ---
--- ✅ Model and tokenizer loaded successfully. ---
--- ✅ Data preprocessing completed. ---
Run name set to: programming-qa-20250629-143022
--- ✅ Weights & Biases setup completed. ---
--- ✅ Trainer initialized successfully. ---
--- ✅ Starting training... ---

Training Progress:
Epoch 1/2: ██████████ 100% | Loss: 1.234
Epoch 2/2: ██████████ 100% | Loss: 0.987

--- ✅ Training completed successfully ---
--- ✅ Model saved to ./models/fine_tuned locally ---
--- ✅ Model uploaded to Hugging Face Hub ---
```

## Step 4: Test Your Fine-Tuned Model

Let's test the model with inference:

### 4.1 Create Test Data

Create a test file `test_questions.jsonl`:

```json
{"question": "What is a loop in programming?", "answer": ""}
{"question": "What is recursion?", "answer": ""}
{"question": "What is a data structure?", "answer": ""}
```

### 4.2 Update Inferencer Configuration

Add this section to your `config.toml`:

```toml
[inferencer]
max_sequence_length = 2048
dtype = "null"
load_in_4bit = true
load_in_8bit = false

# Your test dataset
testing_data_id = "your-username/test-questions"  # Upload test_questions.jsonl first

# Column configuration
question_column = "question"
ground_truth_column = "answer"
system_prompt_column = "null"
system_prompt_override_text = "You are a helpful programming tutor. Provide clear and concise explanations."

# Generation parameters
max_new_tokens = 150
use_cache = true
temperature = 0.7
min_p = 0.1

# Model location
hf_user_id = "your-username"
run_name = "null"  # Will use the latest model
```

### 4.3 Run Inference

```bash
# Upload test dataset first
huggingface-cli upload your-username/test-questions test_questions.jsonl

# Run inference
uv run app/inferencer.py --hf-key "your_hf_token"
```

## Step 5: Evaluate Results

Let's evaluate how well our model performs:

### 5.1 Prepare Ground Truth

Create `ground_truth.jsonl` with expected answers:

```json
{"question": "What is a loop in programming?", "answer": "A loop is a programming construct that repeats a block of code multiple times until a certain condition is met."}
{"question": "What is recursion?", "answer": "Recursion is a programming technique where a function calls itself to solve a problem by breaking it down into smaller, similar subproblems."}
{"question": "What is a data structure?", "answer": "A data structure is a way of organizing and storing data in a computer so that it can be accessed and manipulated efficiently."}
```

### 5.2 Configure Evaluator

Add this to your `config.toml`:

```toml
[evaluator]
# Evaluation metrics
metrics = ["bleu_score", "rouge_score", "semantic_similarity"]

# LLM for evaluation (optional, remove if no OpenAI key)
llm = "gpt-4o-mini"
embedding = "text-embeddings-3-small"

# Run configuration
run_name = "null"
run_name_prefix = "eval-programming-qa-"
run_name_suffix = ""
```

### 5.3 Run Evaluation

```bash
# Run evaluation (with OpenAI key for LLM-based metrics)
uv run app/evaluator.py --openai-key "your_openai_key"

# Or without LLM-based metrics (only BLEU and ROUGE)
uv run app/evaluator.py
```

## Step 6: Review Results

### Training Metrics

1. **Weights & Biases Dashboard**:
   - Go to [wandb.ai](https://wandb.ai)
   - Open your project: `programming-qa-tutorial`
   - Review loss curves, learning rate, and system metrics

2. **Local Model**:
   - Model saved in `./models/fine_tuned/`
   - Hugging Face Hub: `your-username/programming-qa-YYYYMMDD-HHMMSS`

### Inference Results

Check `inferencer_output.jsonl`:

```bash
cat inferencer_output.jsonl
```

Expected output:
```json
{"question": "What is a loop in programming?", "answer": "A loop is a programming construct that allows you to repeat a block of code multiple times...", "metadata": {...}}
```

### Evaluation Metrics

Check the evaluation results:

```bash
# Summary metrics
cat evaluator_output_summary.json

# Detailed results
# Open evaluator_output_detailed.xlsx in Excel/LibreOffice
```

## Understanding the Results

### Training Loss

- **Good Training**: Loss should decrease over epochs
- **Overfitting**: Loss stops decreasing or increases
- **Underfitting**: Loss remains high throughout training

### BLEU Score

- **Range**: 0-100 (higher is better)
- **Good Score**: 20+ for this task
- **Interpretation**: Measures word overlap with reference

### ROUGE Score

- **Range**: 0-1 (higher is better)  
- **Good Score**: 0.2+ for this task
- **Interpretation**: Measures summary quality

### Semantic Similarity

- **Range**: 0-1 (higher is better)
- **Good Score**: 0.7+ for this task
- **Interpretation**: Meaning similarity using embeddings

## Next Steps

Congratulations! You've completed your first fine-tuning project. Here's what you can do next:

### Improve the Model

1. **More Data**: Add more diverse programming questions
2. **Longer Training**: Increase epochs (3-5)
3. **Higher LoRA Rank**: Try rank=32 for better quality
4. **Validation Set**: Add validation data to monitor overfitting

### Advanced Techniques

1. **[Advanced Configuration](advanced-configuration.md)**: Explore more options
2. **[CI/CD Integration](ci-cd-integration.md)**: Automate your pipeline
3. **Multi-GPU Training**: Scale to larger models and datasets

### Experiment with Different Tasks

1. **Text Summarization**: Train on summarization datasets
2. **Code Generation**: Fine-tune for programming tasks
3. **Translation**: Create multilingual models
4. **Classification**: Adapt for classification tasks

## Troubleshooting

### Common Issues

**Out of Memory Error**:
```toml
# Reduce batch size and sequence length
device_train_batch_size = 1
max_sequence_length = 1024
```

**Slow Training**:
```toml
# Enable optimizations
packing = true
use_flash_attention = true
```

**Poor Quality Results**:
```toml
# Increase model capacity
rank = 32
lora_alpha = 32
epochs = 3
```

**Dataset Loading Issues**:
- Check dataset format (JSONL)
- Verify column names match configuration
- Ensure dataset is public or you have access

## Conclusion

You've successfully:

- ✅ Created and uploaded a custom dataset
- ✅ Configured the fine-tuning pipeline
- ✅ Fine-tuned a language model
- ✅ Generated predictions with inference
- ✅ Evaluated model performance

This foundation will help you tackle more complex fine-tuning projects. The same process scales to larger models, datasets, and more sophisticated tasks.
