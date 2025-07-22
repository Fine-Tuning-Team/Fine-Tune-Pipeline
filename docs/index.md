# Fine-Tune Pipeline

Welcome to the **Fine-Tune Pipeline** documentation! This is a comprehensive fine-tuning pipeline for language models that includes training, inference, and evaluation capabilities.

## 🚀 Features

- **Easy Configuration**: TOML-based configuration system for all components
- **Modern Architecture**: Built with Unsloth, Transformers, and TRL for efficient fine-tuning
- **Comprehensive Evaluation**: Multiple evaluation metrics including BLEU, ROUGE, and semantic similarity
- **CI/CD Ready**: Designed to work seamlessly with GitHub Actions and Jenkins
- **Flexible Inference**: Built-in inference capabilities with customizable parameters
- **MLflow Integration**: Comprehensive experiment tracking and model versioning
- **Pipeline Orchestration**: Integrated pipeline runner with phase control

## 🏗️ Architecture

The pipeline consists of four main components:

### 1. **Pipeline Orchestrator** (`app/pipeline_invoker.py`)

- Coordinates the execution of fine-tuning, inference, and evaluation phases
- Comprehensive MLflow experiment tracking and logging
- Configurable phase execution with stop-after options
- Automatic metric sanitization and artifact management

### 2. **Fine-Tuner** (`app/finetuner.py`)

- Handles model fine-tuning using LoRA (Low-Rank Adaptation)
- Supports 4-bit and 8-bit quantization for memory efficiency
- Integrates with Weights & Biases for experiment tracking
- Automatic model publishing to Hugging Face Hub

### 3. **Inferencer** (`app/inferencer.py`)

- Performs inference on test datasets
- Configurable generation parameters
- Supports both local and Hub models
- Outputs results in JSONL format

### 4. **Evaluator** (`app/evaluator.py`)

- Comprehensive evaluation suite with multiple metrics
- Support for both traditional (BLEU, ROUGE) and LLM-based evaluation
- Detailed reporting with Excel and JSON outputs
- Semantic similarity and factual correctness evaluation

### 5. **Documentation Server** (`app/docs_server.py`)

- Built-in documentation server with MkDocs integration
- Automatic dependency management (uv/pip fallback)
- Local development server for documentation
- Static documentation build capabilities

## 🛠️ Technology Stack

- **[Unsloth](https://github.com/unslothai/unsloth)**: Efficient fine-tuning framework
- **[Transformers](https://huggingface.co/transformers/)**: Hugging Face Transformers library
- **[TRL](https://github.com/huggingface/trl)**: Transformer Reinforcement Learning
- **[Datasets](https://huggingface.co/docs/datasets/)**: Hugging Face Datasets library
- **[MLflow](https://mlflow.org/)**: Machine learning lifecycle management and experiment tracking
- **[Weights & Biases](https://wandb.ai/)**: Additional experiment tracking and logging
- **[RAGAS](https://github.com/explodinggradients/ragas)**: Retrieval Augmented Generation Assessment

## 📊 Supported Models

The pipeline supports various model architectures including:

- Qwen 2.5 series
- Llama models
- Mistral models
- And any model compatible with Unsloth

## 📈 Evaluation Metrics

Built-in support for:

- **BLEU Score**: Translation quality assessment
- **ROUGE Score**: Summarization evaluation
- **Factual Correctness**: LLM-based factual evaluation
- **Semantic Similarity**: Embedding-based similarity
- **Answer Accuracy**: Custom accuracy metrics
- **Answer Relevancy**: Relevance assessment

## 🔧 Quick Start

Ready to get started? Check out our [Quick Start Guide](getting-started/quick-start.md) to begin fine-tuning your first model!

### Pipeline Execution

```bash
# Run complete pipeline with MLflow tracking
python app/pipeline_invoker.py --config config.toml --hf-key YOUR_HF_TOKEN --openai-key YOUR_OPENAI_KEY

# Run specific phases
python app/pipeline_invoker.py --enable-finetuning --enable-inference --enable-evaluation

# Start documentation server
python app/docs_server.py
```

## 📚 What's Next?

- [Installation Guide](getting-started/installation.md) - Set up your environment
- [Configuration Overview](configuration/overview.md) - Understand the configuration system
- [Basic Fine-Tuning Tutorial](tutorials/basic-fine-tuning.md) - Your first fine-tuning project
- [API Reference](api-reference.md) - Detailed API documentation

---

*For issues or contributions, please visit our [GitHub repository](https://github.com/Fine-Tuning-Team/Fine-Tune-Pipeline).*
