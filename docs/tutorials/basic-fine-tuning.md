# Basic Fine-Tuning Tutorial

This tutorial will guide you through your first fine-tuning project using the Fine-Tune Pipeline. We'll fine-tune a small language model on a question-answering dataset about programming concepts.

## Prerequisites

Before starting this tutorial, ensure you have:

- ✅ [Set up your environment](../getting-started/environment-setup.md)
- ✅ API keys for Hugging Face, OpenAI, and optionally Weights & Biases
- ✅ Access to a GitHub repository with the Fine-Tune Pipeline

## Tutorial Overview

In this tutorial, we will:

1. Prepare custom datasets (training, testing, and ground truth)
2. Configure the complete pipeline (fine-tuning, inference, and evaluation)
3. Trigger the automated pipeline execution
4. Monitor and review results

**Estimated Time**: 45-60 minutes (including pipeline execution)

## Step 1: Prepare Your Datasets

We'll create three datasets for the complete pipeline:

### 1.1 Training Dataset

Create a file called `programming_qa.jsonl`:

```json
{"question": "What is a variable in programming?", "answer": "A variable is a named storage location in memory that holds a value which can be modified during program execution."}
{"question": "What is a function?", "answer": "A function is a reusable block of code that performs a specific task and can accept inputs (parameters) and return outputs."}
{"question": "What is object-oriented programming?", "answer": "Object-oriented programming (OOP) is a programming paradigm based on the concept of objects, which contain data (attributes) and code (methods)."}
{"question": "What is an algorithm?", "answer": "An algorithm is a step-by-step procedure or set of instructions designed to solve a specific problem or perform a particular task."}
{"question": "What is debugging?", "answer": "Debugging is the process of finding, analyzing, and fixing errors or bugs in computer programs to ensure they work correctly."}
{"question": "What is a loop?", "answer": "A loop is a programming construct that repeats a block of code multiple times until a certain condition is met."}
{"question": "What is recursion?", "answer": "Recursion is a programming technique where a function calls itself to solve a problem by breaking it down into smaller, similar subproblems."}
{"question": "What is a data structure?", "answer": "A data structure is a way of organizing and storing data in a computer so that it can be accessed and manipulated efficiently."}
```

### 1.2 Test Dataset (for Inference)

Create a file called `test_questions.jsonl`:

```json
{"question": "What is a while loop in programming?", "answer": ""}
{"question": "Explain the concept of inheritance in OOP.", "answer": ""}
{"question": "What is the difference between a stack and a queue?", "answer": ""}
{"question": "What is binary search?", "answer": ""}
```

### 1.3 Ground Truth Dataset (for Evaluation)

Create a file called `ground_truth.jsonl`:

```json
{"question": "What is a while loop in programming?", "answer": "A while loop is a control flow statement that repeatedly executes a block of code as long as a specified condition remains true."}
{"question": "Explain the concept of inheritance in OOP.", "answer": "Inheritance is a fundamental concept in object-oriented programming that allows a class to inherit properties and methods from another class, promoting code reuse and establishing relationships between classes."}
{"question": "What is the difference between a stack and a queue?", "answer": "A stack follows Last-In-First-Out (LIFO) principle where elements are added and removed from the same end, while a queue follows First-In-First-Out (FIFO) principle where elements are added at one end and removed from the other."}
{"question": "What is binary search?", "answer": "Binary search is an efficient algorithm for finding a target value in a sorted array by repeatedly dividing the search interval in half and comparing the target with the middle element."}
```

### 1.4 Upload Datasets to Hugging Face Hub

1. **Create dataset repositories** on [Hugging Face Hub](https://huggingface.co/new-dataset):
   - `your-username/programming-qa-training`
   - `your-username/programming-qa-testing`
   - `your-username/programming-qa-ground-truth`

2. **Upload your datasets** using the web interface or CLI:

```bash
# If using Hugging Face CLI locally
huggingface-cli login

# Upload training dataset
huggingface-cli repo create your-username/programming-qa-training --type dataset
huggingface-cli upload your-username/programming-qa-training programming_qa.jsonl

# Upload test dataset
huggingface-cli repo create your-username/programming-qa-testing --type dataset
huggingface-cli upload your-username/programming-qa-testing test_questions.jsonl

# Upload ground truth dataset
huggingface-cli repo create your-username/programming-qa-ground-truth --type dataset
huggingface-cli upload your-username/programming-qa-ground-truth ground_truth.jsonl
```

## Step 2: Configure the Complete Pipeline

Now we'll configure all three components (fine-tuning, inference, and evaluation) in a single `config.toml` file:

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

# Training dataset
training_data_id = "your-username/programming-qa-training"
validation_data_id = "null"  # No validation set for this tutorial

# Dataset configuration
dataset_num_proc = 2
question_column = "question"
ground_truth_column = "answer"
system_prompt_column = "null"
system_prompt_override_text = "You are a helpful programming tutor. Provide clear and concise explanations."

# Training parameters - short training for tutorial
epochs = 3
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

[inferencer]
# Model configuration
max_sequence_length = 2048
dtype = "null"
load_in_4bit = true
load_in_8bit = false

# Test dataset for inference
testing_data_id = "your-username/programming-qa-testing"

# Column configuration
question_column = "question"
ground_truth_column = "answer"
system_prompt_column = "null"
system_prompt_override_text = "You are a helpful programming tutor. Provide clear and concise explanations."

# Generation parameters
max_new_tokens = 200
use_cache = true
temperature = 0.7
min_p = 0.1

# Model location - will use the model just fine-tuned
hf_user_id = "your-username"
run_name = "null"  # Will use the latest model from fine-tuning

[evaluator]
# Ground truth dataset for evaluation
ground_truth_data_id = "your-username/programming-qa-ground-truth"

# Evaluation metrics
metrics = ["bleu_score", "rouge_score", "semantic_similarity", "factual_correctness"]

# LLM for advanced evaluation (requires OpenAI API key)
llm = "gpt-4o-mini"
embedding = "text-embeddings-3-small"

# Column configuration
question_column = "question"
ground_truth_column = "answer"
prediction_column = "answer"

# Run configuration
run_name = "null"
run_name_prefix = "eval-programming-qa-"
run_name_suffix = ""

# MLflow tracking
mlflow_experiment_name = "programming-qa-evaluation"
```

### Configuration Explanation

| Section | Key Parameters | Purpose |
|---------|---------------|---------|
| **Fine-Tuner** | `base_model_id`, `epochs`, `rank` | Defines the model and training process |
| **Inferencer** | `testing_data_id`, `max_new_tokens` | Configures inference on test data |
| **Evaluator** | `ground_truth_data_id`, `metrics` | Sets up evaluation against ground truth |

## Step 3: Trigger the Complete Pipeline

With all configurations in place, commit and push to trigger the automated pipeline:

```bash
# Add the updated config file
git add config.toml

# Commit with a descriptive message
git commit -m "Add complete pipeline config for programming QA tutorial"

# Push to trigger GitHub Actions
git push origin your-branch-name
```

## Step 4: Monitor Pipeline Execution

The pipeline will automatically execute in the following sequence:

### 4.1 Fine-Tuning Phase

- **Duration**: ~15-20 minutes
- **Process**: Downloads model, trains on your dataset, uploads to Hub
- **Output**: Fine-tuned model in `your-username/programming-qa-YYYYMMDD-HHMMSS`

### 4.2 Inference Phase

- **Duration**: ~5-10 minutes  
- **Process**: Loads fine-tuned model, generates answers for test questions
- **Output**: Inference results in JSONL format

### 4.3 Evaluation Phase

- **Duration**: ~5-10 minutes
- **Process**: Compares predictions with ground truth using multiple metrics
- **Output**: Evaluation report with scores and analysis

### 4.4 Monitoring Progress

1. **GitHub Actions**: Go to the "Actions" tab in your repository
2. **Real-time Logs**: Click on the running workflow to see live progress
3. **Phase Indicators**: Look for phase completion messages:

   ```sh
   --- ✅ Fine-tuning completed successfully ---
   --- ✅ Inference completed successfully ---
   --- ✅ Evaluation completed successfully ---
   ```

## Step 5: Review Results

### 5.1 Training Metrics

**MLFlow Dashboard**:

- Go to [mlflow server](https://33008a58-e51f-4442-994c-c4841203c6fb.e1-us-east-azure.choreoapps.dev/) 
- Project: `programming-qa-tutorial`
- Review: Loss curves, learning rate schedules, GPU utilization

**Expected Training Progress**:

```text
Epoch 1/3: Loss: 1.234 → 0.987
Epoch 2/3: Loss: 0.987 → 0.756  
Epoch 3/3: Loss: 0.756 → 0.612
```

### 5.2 Inference Results

**Location**: Automatically uploaded to Hugging Face Hub
**Format**: JSONL file with generated answers
**Sample**:

```json
{"question": "What is a while loop in programming?", "answer": "A while loop is a control structure that repeatedly executes code as long as a condition is true..."}
```

### 5.3 Evaluation Metrics

**MLflow Dashboard**: 

- URL: [MLflow Server](https://33008a58-e51f-4442-994c-c4841203c6fb.e1-us-east-azure.choreoapps.dev/)
- Experiment: `programming-qa-evaluation`

**Key Metrics to Review**:

| Metric | Good Score | Interpretation |
|--------|------------|----------------|
| **BLEU Score** | 0.3+ | Word-level similarity with reference |
| **ROUGE-L** | 0.4+ | Longest common subsequence overlap |
| **Semantic Similarity** | 0.6+ | Vector similarity using embeddings |
| **Factual Correctness** | 0.7+ | LLM-judged factual accuracy |

### 5.4 Detailed Results Files

All results are automatically uploaded to Hugging Face Hub:

- **Training logs**: Model repository
- **Inference output**: `inferencer_output.jsonl`
- **Evaluation report**: `evaluator_output_detailed.xlsx`
- **Summary metrics**: `evaluator_output_summary.json`

## Understanding Your Results

### Good Performance Indicators

- ✅ Training loss decreases consistently
- ✅ BLEU score > 0.3  
- ✅ Semantic similarity > 0.6
- ✅ Generated answers are coherent and relevant

### Warning Signs

- ⚠️ Training loss plateaus early (possible underfitting)
- ⚠️ Very low BLEU scores < 0.1 (poor word overlap)
- ⚠️ Generated answers are repetitive or off-topic

### Next Steps Based on Results

**If results are good (metrics above thresholds)**:

- Experiment with larger models
- Add more diverse training data
- Try longer training (more epochs)

**If results need improvement**:

- Increase LoRA rank (16 → 32)
- Add more training data
- Adjust generation parameters (temperature, max_tokens)
- Use response-only training for better instruction following

## Advanced Improvements

### 1. Scale Up Training

```toml
[fine_tuner]
epochs = 40                   # Longer training
rank = 32                      # Higher LoRA capacity
device_train_batch_size = 4    # Larger batches (if GPU allows)
```

### 2. Improve Generation Quality

```toml
[inferencer]
temperature = 0.3              # More focused generation
max_new_tokens = 3000           # Longer responses
```

### 3. Add More Evaluation Metrics

```toml
[evaluator]
metrics = ["bleu_score", "rouge_score", "semantic_similarity", 
          "factual_correctness", "answer_accuracy", "coherence"]
```

## Troubleshooting

### Common Issues

**Pipeline fails at fine-tuning**:

- Check dataset format and column names
- Verify Hugging Face token has write permissions
- Ensure model ID is correct

**Low evaluation scores**:

- Increase training epochs or LoRA rank
- Check if ground truth answers are high quality
- Verify question-answer alignment in datasets

**Out of memory errors**:

- Reduce batch size: `device_train_batch_size = 1`
- Reduce sequence length: `max_sequence_length = 1024`
- Enable gradient checkpointing

## Conclusion

Congratulations! You've successfully:

- ✅ Created a complete dataset pipeline (training, testing, ground truth)
- ✅ Configured an end-to-end fine-tuning pipeline
- ✅ Triggered automated execution via GitHub Actions
- ✅ Monitored training, inference, and evaluation phases
- ✅ Reviewed comprehensive results and metrics

This workflow scales to any domain - just replace the datasets and adjust the configurations. The same process works for:

- Text summarization
- Code generation  
- Translation tasks
- Classification problems

Ready for more advanced techniques? Check out:

- [Advanced Configuration](advanced-configuration.md)
- [CI/CD Integration](ci-cd-integration.md)
