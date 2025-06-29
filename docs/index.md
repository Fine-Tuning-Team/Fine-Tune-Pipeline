# Fine-Tune Pipeline

Welcome to the **Fine-Tune Pipeline** documentation! This is a comprehensive fine-tuning pipeline for language models that includes training, inference, and evaluation capabilities.

## 🚀 Features

- **Easy Configuration**: TOML-based configuration system for all components
- **Modern Architecture**: Built with Unsloth, Transformers, and TRL for efficient fine-tuning
- **Comprehensive Evaluation**: Multiple evaluation metrics including BLEU, ROUGE, and semantic similarity
- **CI/CD Ready**: Designed to work seamlessly with GitHub Actions and Jenkins
- **Flexible Inference**: Built-in inference capabilities with customizable parameters
- **Weights & Biases Integration**: Automatic logging and experiment tracking

## 🏗️ Architecture

The pipeline consists of three main components:

### 1. **Fine-Tuner** (`app/finetuner.py`)

- Handles model fine-tuning using LoRA (Low-Rank Adaptation)
- Supports 4-bit and 8-bit quantization for memory efficiency
- Integrates with Weights & Biases for experiment tracking
- Automatic model publishing to Hugging Face Hub

### 2. **Inferencer** (`app/inferencer.py`)

- Performs inference on test datasets
- Configurable generation parameters
- Supports both local and Hub models
- Outputs results in JSONL format

### 3. **Evaluator** (`app/evaluator.py`)

- Comprehensive evaluation suite with multiple metrics
- Support for both traditional (BLEU, ROUGE) and LLM-based evaluation
- Detailed reporting with Excel and JSON outputs
- Semantic similarity and factual correctness evaluation

## 🛠️ Technology Stack

- **[Unsloth](https://github.com/unslothai/unsloth)**: Efficient fine-tuning framework
- **[Transformers](https://huggingface.co/transformers/)**: Hugging Face Transformers library
- **[TRL](https://github.com/huggingface/trl)**: Transformer Reinforcement Learning
- **[Datasets](https://huggingface.co/docs/datasets/)**: Hugging Face Datasets library
- **[Weights & Biases](https://wandb.ai/)**: Experiment tracking and logging
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

## 📚 What's Next?

- [Installation Guide](getting-started/installation.md) - Set up your environment
- [Configuration Overview](configuration/overview.md) - Understand the configuration system
- [Basic Fine-Tuning Tutorial](tutorials/basic-fine-tuning.md) - Your first fine-tuning project
- [API Reference](api-reference.md) - Detailed API documentation

---

*This documentation is automatically generated and maintained. For issues or contributions, please visit our [GitHub repository](https://github.com/your-username/Fine-Tune-Pipeline).*
