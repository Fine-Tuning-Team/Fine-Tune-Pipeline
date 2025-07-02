# Advanced Configuration

This tutorial covers advanced configuration options and techniques for optimizing your Fine-Tune Pipeline for specific use cases and performance requirements.

## Prerequisites

Before diving into advanced configuration, ensure you have:

- ✅ Completed the [Basic Fine-Tuning Tutorial](basic-fine-tuning.md)
- ✅ Understanding of your hardware limitations
- ✅ Familiarity with your specific use case requirements
- ✅ Access to validation datasets for optimization

## Advanced LoRA Configuration

### High-Rank LoRA for Maximum Quality

For maximum adaptation capacity at the cost of more parameters:

```toml
[fine_tuner]
# High-capacity LoRA setup
rank = 128
lora_alpha = 64
lora_dropout = 0.05

# Target more modules for comprehensive adaptation
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",          # Attention
    "gate_proj", "up_proj", "down_proj",             # MLP
    "embed_tokens", "lm_head"                        # Embeddings
]

# Advanced LoRA techniques
use_rslora = true     # Rank-Stabilized LoRA
```

### LoFTQ Integration

LoFTQ (LoRA-Fine-Tuning-aware Quantization) for better quantized fine-tuning:

```toml
[fine_tuner]
rank = 32
lora_alpha = 32

# LoFTQ configuration
loftq_config = {
    "loftq_bits": 4,
    "loftq_iter": 1
}

# Must use with quantization
load_in_4bit = true
```

### Dynamic LoRA Scaling

```toml
[fine_tuner]
# Start with lower rank, increase if needed
rank = 16
lora_alpha = 32  # Higher alpha for stronger adaptation

# Use dropout scheduling
lora_dropout = 0.1  # Start higher, reduce during training
```

## Memory Optimization Strategies

### Extreme Memory Constraints (4GB GPU)

```toml
[fine_tuner]
# Ultra-efficient setup
base_model_id = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
max_sequence_length = 512
load_in_4bit = true

# Minimal batch processing
device_train_batch_size = 1
grad_accumulation = 32  # Maintain effective batch size of 32

# Memory-saving techniques
use_gradient_checkpointing = "unsloth"
packing = false  # Disable for memory savings
dataset_num_proc = 1  # Reduce parallel processing

# Conservative LoRA
rank = 8
lora_alpha = 16
target_modules = ["q_proj", "v_proj"]  # Minimal targets
```

### High-Memory Systems (24GB+ GPU)

```toml
[fine_tuner]
# Take advantage of available memory
base_model_id = "unsloth/Llama-3.1-8B-Instruct-bnb-4bit"
max_sequence_length = 8192
device_train_batch_size = 8
grad_accumulation = 2

# High-capacity LoRA
rank = 64
lora_alpha = 32
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "embed_tokens", "lm_head"
]

# Performance optimizations
packing = true
use_flash_attention = true
dataset_num_proc = 8
```

## Training Optimization

### Learning Rate Scheduling

```toml
[fine_tuner]
# Sophisticated learning rate schedule
learning_rate = 0.0003
warmup_steps = 100
lr_scheduler_type = "cosine"

# Longer training with decay
epochs = 5
weight_decay = 0.01

# Monitor for overfitting
save_steps = 50
eval_steps = 50  # If validation set available
```

### Curriculum Learning

Start with easier examples, progress to harder ones:

```toml
[fine_tuner]
# Phase 1: Easy examples (short, simple)
training_data_id = "username/easy-examples"
epochs = 2
learning_rate = 0.0005

# Phase 2: Medium difficulty
# Update config and continue training
training_data_id = "username/medium-examples"
epochs = 2
learning_rate = 0.0002

# Phase 3: Full dataset
training_data_id = "username/full-dataset"
epochs = 1
learning_rate = 0.0001
```

### Multi-Stage Training

```toml
# Stage 1: Quick adaptation
[fine_tuner]
rank = 16
learning_rate = 0.0005
epochs = 1

# Stage 2: Quality refinement
# rank = 32  # Increase capacity
# learning_rate = 0.0001
# epochs = 3
```

## Data Engineering

### Advanced Data Processing

```toml
[fine_tuner]
# Sophisticated prompt engineering
system_prompt_override_text = """You are an expert assistant with the following capabilities:
1. Provide accurate, well-researched information
2. Cite sources when applicable
3. Admit uncertainty when appropriate
4. Use clear, concise language

Instructions: {task_specific_instructions}
Context: {domain_context}"""

# Dynamic prompt templates
question_part = "<|im_start|>user\n{context}\n\nQuestion: "
answer_part = "<|im_start|>assistant\nLet me think about this step by step.\n\n"
```

### Data Augmentation

Create variations of your training data:

```python
# Example data augmentation script
import random

def augment_qa_pair(question, answer):
    """Create variations of Q&A pairs."""
    
    # Question variations
    question_templates = [
        f"Can you explain {question.lower()}?",
        f"What is your understanding of {question.lower()}?",
        f"Please describe {question.lower()}.",
        f"Help me understand {question.lower()}."
    ]
    
    # Answer style variations
    answer_styles = [
        f"Certainly! {answer}",
        f"Here's my explanation: {answer}",
        f"Let me break this down: {answer}",
        f"To answer your question: {answer}"
    ]
    
    return random.choice(question_templates), random.choice(answer_styles)
```

### Quality Filtering

```python
def filter_high_quality_samples(dataset, min_length=50, max_length=1000):
    """Filter dataset for quality samples."""
    
    filtered = []
    for sample in dataset:
        question = sample['question']
        answer = sample['answer']
        
        # Length filters
        if not (min_length <= len(answer) <= max_length):
            continue
            
        # Quality heuristics
        if answer.count('.') < 2:  # Too short/simple
            continue
            
        if question.lower() in answer.lower():  # Repetitive
            continue
            
        # Language quality
        if not is_coherent_text(answer):
            continue
            
        filtered.append(sample)
    
    return filtered
```

## Multi-GPU Training

### Data Parallel Training

```toml
[fine_tuner]
# Configure for multi-GPU setup
device_train_batch_size = 4  # Per GPU
grad_accumulation = 2        # Per GPU

# Total effective batch size = num_gpus * batch_size * grad_accumulation
# Example: 2 GPUs * 4 batch * 2 accumulation = 16 effective batch size

# Optimize for multi-GPU
dataset_num_proc = 16  # More parallel processing
save_steps = 100       # Less frequent saves
```

### Mixed Precision Training

```toml
[fine_tuner]
# Automatic mixed precision
dtype = "null"  # Auto-select best precision

# Manual precision control
# dtype = "bfloat16"  # For A100, H100
# dtype = "float16"   # For older GPUs
```

## Domain-Specific Optimizations

### Code Generation

```toml
[fine_tuner]
# Optimized for code tasks
base_model_id = "unsloth/CodeLlama-7B-Instruct-bnb-4bit"
max_sequence_length = 4096  # Longer for code

# Code-specific prompting
system_prompt_override_text = """You are an expert programmer. Provide:
1. Clean, well-commented code
2. Explanation of logic
3. Best practices and conventions
4. Error handling where appropriate"""

# Training parameters for code
learning_rate = 0.0001  # Lower for code precision
epochs = 3
train_on_responses_only = true  # Focus on code generation
```

### Mathematical Reasoning

```toml
[fine_tuner]
# Math-focused setup
system_prompt_override_text = """You are a mathematics tutor. Always:
1. Show step-by-step solutions
2. Explain mathematical concepts clearly
3. Check your work for accuracy
4. Use proper mathematical notation"""

# Longer sequences for detailed explanations
max_sequence_length = 2048
max_new_tokens = 512

# Conservative training to maintain accuracy
learning_rate = 0.00005
epochs = 2
```

### Conversational AI

```toml
[fine_tuner]
# Optimized for dialogue
system_prompt_override_text = """You are a helpful, empathetic assistant. You:
1. Listen carefully to user concerns
2. Provide thoughtful, personalized responses
3. Ask clarifying questions when needed
4. Maintain context throughout conversations"""

# Dialogue-specific parameters
temperature = 0.8           # More creative responses
repetition_penalty = 1.2    # Avoid repetitive responses
train_on_responses_only = true
```

## Validation and Monitoring

### Advanced Validation Setup

```toml
[fine_tuner]
# Comprehensive validation
validation_data_id = "username/validation-set"
eval_steps = 25             # Frequent evaluation
save_strategy = "steps"
evaluation_strategy = "steps"

# Early stopping
load_best_model_at_end = true
metric_for_best_model = "eval_loss"
greater_is_better = false
```

### Custom Metrics Monitoring

```python
class CustomCallback:
    def on_evaluate(self, logs):
        """Custom evaluation callback."""
        
        # Custom metrics
        perplexity = math.exp(logs.get('eval_loss', 0))
        logs['eval_perplexity'] = perplexity
        
        # Log to Weights & Biases
        wandb.log({
            'eval/perplexity': perplexity,
            'eval/custom_score': self.compute_custom_score()
        })
```

## Performance Monitoring

### Comprehensive Logging

```toml
[fine_tuner]
# Detailed logging setup
log_steps = 5
logging_first_step = true
report_to = "wandb"

# Custom logging configuration
wandb_project_name = "advanced-fine-tuning"
run_name_prefix = "exp-"

# Additional metrics
dataloader_drop_last = false
include_inputs_for_metrics = true
```

### Resource Monitoring

```python
import psutil
import GPUtil

def monitor_resources():
    """Monitor system resources during training."""
    
    # CPU and Memory
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    # GPU monitoring
    gpus = GPUtil.getGPUs()
    gpu_stats = {
        f'gpu_{i}_utilization': gpu.load * 100,
        f'gpu_{i}_memory': gpu.memoryUtil * 100,
        f'gpu_{i}_temperature': gpu.temperature
        for i, gpu in enumerate(gpus)
    }
    
    # Log to Weights & Biases
    wandb.log({
        'system/cpu_percent': cpu_percent,
        'system/memory_percent': memory.percent,
        **gpu_stats
    })
```

## Troubleshooting Advanced Setups

### Memory Issues

```bash
# Monitor GPU memory usage
nvidia-smi -l 1

# Check for memory leaks
python -c "
import torch
print(f'Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
print(f'Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB')
"
```

### Training Instability

```toml
[fine_tuner]
# Stabilize training
learning_rate = 0.00005      # Lower learning rate
weight_decay = 0.01          # L2 regularization
grad_clip_norm = 1.0         # Gradient clipping

# Use warmup
warmup_steps = 100
lr_scheduler_type = "linear"
```

### Convergence Issues

```toml
[fine_tuner]
# Improve convergence
epochs = 10                  # More training
eval_steps = 50             # Frequent evaluation
patience = 3                # Early stopping patience

# Data quality
max_seq_length = 2048       # Consistent length
packing = false             # Avoid sequence mixing
```

## Production Considerations

### Model Versioning

```toml
[fine_tuner]
# Systematic versioning
run_name_prefix = "v2-production-"
push_to_hub = true
save_total_limit = 5

# Detailed metadata
tags = ["production", "v2.0", "optimized"]
```

### Reproducibility

```toml
[fine_tuner]
# Ensure reproducibility
seed = 42
deterministic = true

# Version control
log_model_config = true
save_training_args = true
```

### Automated Hyperparameter Tuning

```python
import optuna

def objective(trial):
    """Optuna objective for hyperparameter optimization."""
    
    # Suggest hyperparameters
    rank = trial.suggest_int('rank', 8, 64, step=8)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    epochs = trial.suggest_int('epochs', 1, 5)
    
    # Update configuration
    config.rank = rank
    config.learning_rate = learning_rate
    config.epochs = epochs
    
    # Run training
    tuner = FineTune(config=config)
    stats = tuner.run()
    
    # Return metric to optimize
    return stats.eval_loss

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
```

## Next Steps

After mastering advanced configuration:

1. **Experiment Tracking**: Set up comprehensive experiment management
2. **A/B Testing**: Compare different configurations systematically
3. **Production Deployment**: Scale your optimized models
4. **Continuous Learning**: Implement online learning workflows

Your advanced configuration skills will enable you to squeeze maximum performance from your fine-tuning pipeline while efficiently managing computational resources.
