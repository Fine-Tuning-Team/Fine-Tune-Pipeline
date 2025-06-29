# Evaluator Configuration

The `[evaluator]` section controls the model evaluation process, which measures the quality of your fine-tuned model's outputs using various metrics.

## Basic Configuration

```toml
[evaluator]
# Evaluation metrics to compute
metrics = ["bleu_score", "rouge_score", "semantic_similarity"]

# LLM for advanced evaluation (optional)
llm = "gpt-4o-mini"
embedding = "text-embeddings-3-small"

# Run configuration
run_name = "null"
run_name_prefix = "eval-"
run_name_suffix = ""
```

## Supported Metrics

### Traditional Metrics

```toml
[evaluator]
metrics = [
    "bleu_score",        # Translation quality
    "rouge_score",       # Summarization quality
]
```

**BLEU Score**:
- Range: 0-100 (higher is better)
- Good for: Translation, text generation
- Measures: Word overlap with reference

**ROUGE Score**:
- Range: 0-1 (higher is better)  
- Good for: Summarization, paraphrasing
- Measures: N-gram recall

### LLM-Based Metrics

```toml
[evaluator]
metrics = [
    "factual_correctness",  # Fact verification
    "semantic_similarity",  # Meaning similarity
    "answer_accuracy",      # Answer correctness
    "answer_relevancy",     # Response relevance
    "answer_correctness"    # Overall correctness
]

# Required for LLM-based metrics
llm = "gpt-4o-mini"
embedding = "text-embeddings-3-small"
```

**Factual Correctness**:
- Uses LLM to verify factual accuracy
- Good for: QA, factual content
- Expensive but highly accurate

**Semantic Similarity**:
- Uses embeddings to measure meaning similarity
- Good for: Paraphrasing, content similarity
- Fast and cost-effective

## LLM Configuration

### OpenAI Models

```toml
[evaluator]
# Recommended models
llm = "gpt-4o-mini"      # Cost-effective, good quality
# llm = "gpt-4o"         # Higher quality, more expensive
# llm = "gpt-3.5-turbo"  # Fastest, lower cost

# Embedding models
embedding = "text-embeddings-3-small"  # Recommended
# embedding = "text-embeddings-3-large" # Higher quality
```

### Cost Considerations

| Model | Cost/1K tokens | Use Case |
|-------|----------------|----------|
| gpt-4o-mini | $0.15/$0.60 | Development, large-scale evaluation |
| gpt-4o | $5.00/$15.00 | Production, highest quality |
| gpt-3.5-turbo | $0.50/$1.50 | Budget-conscious evaluation |

## Run Configuration

### Naming Convention

```toml
[evaluator]
# Custom run name
run_name = "experiment-v1"

# Or auto-generated with prefix/suffix
run_name = "null"
run_name_prefix = "eval-qa-"
run_name_suffix = "-final"
# Results in: eval-qa-20250629-143022-final
```

## Input Data Format

The evaluator expects:

1. **Predictions file**: `inferencer_output.jsonl`
2. **Ground truth data**: From your test dataset

### Expected Format

```json
{
    "question": "What is machine learning?",
    "predicted_answer": "Machine learning is a subset of AI...",
    "ground_truth": "ML is a method of data analysis...",
    "metadata": {...}
}
```

## Output Files

The evaluator generates:

### 1. Summary Report (`evaluator_output_summary.json`)

```json
{
    "overall_scores": {
        "bleu_score": 25.4,
        "rouge_score": 0.68,
        "semantic_similarity": 0.82
    },
    "metadata": {
        "total_samples": 100,
        "evaluation_time": "2023-06-29T14:30:22",
        "metrics_used": ["bleu_score", "rouge_score"]
    }
}
```

### 2. Detailed Report (`evaluator_output_detailed.xlsx`)

Excel file with:
- Individual scores per sample
- Question-answer pairs
- Metric breakdowns
- Statistical analysis

## Performance Configuration

### Speed Optimization

```toml
[evaluator]
# Use only fast metrics
metrics = ["bleu_score", "rouge_score"]

# Skip LLM-based evaluation
llm = "null"
```

### Quality Optimization

```toml
[evaluator]
# Use comprehensive metrics
metrics = [
    "bleu_score", 
    "rouge_score", 
    "factual_correctness",
    "semantic_similarity",
    "answer_accuracy"
]

# Use high-quality models
llm = "gpt-4o"
embedding = "text-embeddings-3-large"
```

## Usage Examples

### Basic Evaluation

```bash
# Evaluate with traditional metrics only
uv run app/evaluator.py
```

### Full Evaluation with LLM

```bash
# Include LLM-based metrics
uv run app/evaluator.py --openai-key "your_openai_key"
```

### Programmatic Usage

```python
from app.evaluator import Evaluator

# Initialize evaluator
evaluator = Evaluator()

# Run evaluation
results = evaluator.run()
print(f"Overall BLEU score: {results['bleu_score']}")
```

## Metric Interpretation

### BLEU Score Guidelines

| Score Range | Quality | Interpretation |
|-------------|---------|----------------|
| 0-10 | Poor | Almost no overlap with reference |
| 10-20 | Fair | Some overlap, needs improvement |
| 20-40 | Good | Reasonable quality |
| 40+ | Excellent | High-quality generation |

### ROUGE Score Guidelines

| Score Range | Quality | Interpretation |
|-------------|---------|----------------|
| 0-0.2 | Poor | Low recall of important content |
| 0.2-0.4 | Fair | Moderate content coverage |
| 0.4-0.6 | Good | Good content recall |
| 0.6+ | Excellent | High content overlap |

### Semantic Similarity Guidelines

| Score Range | Quality | Interpretation |
|-------------|---------|----------------|
| 0-0.5 | Poor | Different meaning |
| 0.5-0.7 | Fair | Related but different |
| 0.7-0.85 | Good | Similar meaning |
| 0.85+ | Excellent | Nearly identical meaning |

## Configuration Examples

### Research Evaluation

```toml
[evaluator]
# Comprehensive metrics for research
metrics = [
    "bleu_score",
    "rouge_score", 
    "factual_correctness",
    "semantic_similarity",
    "answer_accuracy",
    "answer_relevancy"
]

llm = "gpt-4o"
embedding = "text-embeddings-3-large"
run_name_prefix = "research-"
```

### Production Monitoring

```toml
[evaluator]
# Fast metrics for regular monitoring
metrics = ["bleu_score", "semantic_similarity"]

llm = "gpt-4o-mini"
embedding = "text-embeddings-3-small"
run_name_prefix = "prod-monitor-"
```

### Budget-Conscious Evaluation

```toml
[evaluator]
# Traditional metrics only (no API costs)
metrics = ["bleu_score", "rouge_score"]

llm = "null"  # Disable LLM-based metrics
embedding = "null"
```

## Error Handling

### API Key Issues

```bash
# Error: OpenAI API key not provided
export OPENAI_API_KEY="your_key"
# or
uv run app/evaluator.py --openai-key "your_key"
```

### Rate Limiting

```toml
[evaluator]
# Use cheaper model to avoid rate limits
llm = "gpt-4o-mini"

# Or disable LLM metrics
metrics = ["bleu_score", "rouge_score"]
```

### Data Format Issues

Ensure your data matches the expected format:
- Questions and answers are strings
- Ground truth is available
- No missing or null values in required fields
