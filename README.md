# Fine-Tune Pipeline

A comprehensive fine-tuning pipeline for language models with evaluation and inference capabilities, designed for use with GitHub Actions or Jenkins.

## 🚀 Features

- **Easy Configuration**: TOML-based configuration system
- **Modern Architecture**: Built with Unsloth, Transformers, and TRL
- **Comprehensive Evaluation**: Multiple evaluation metrics
- **CI/CD Ready**: GitHub Actions and Jenkins integration
- **Memory Efficient**: 4-bit/8-bit quantization support
- **Experiment Tracking**: Weights & Biases integration

## 📚 Documentation

Complete documentation is available at: **[https://your-username.github.io/Fine-Tune-Pipeline](https://your-username.github.io/Fine-Tune-Pipeline)**

### Quick Links

- [Installation Guide](https://your-username.github.io/Fine-Tune-Pipeline/getting-started/installation/)
- [Quick Start Tutorial](https://your-username.github.io/Fine-Tune-Pipeline/getting-started/quick-start/)
- [Configuration Reference](https://your-username.github.io/Fine-Tune-Pipeline/configuration/overview/)
- [API Documentation](https://your-username.github.io/Fine-Tune-Pipeline/api-reference/)

## 🏃 Quick Start

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Set up API keys** (see [Environment Setup](https://your-username.github.io/Fine-Tune-Pipeline/getting-started/environment-setup/)):
   ```bash
   export HF_TOKEN="your_hf_token"
   export WANDB_TOKEN="your_wandb_key"
   ```

3. **Run fine-tuning**:
   ```bash
   uv run app/finetuner.py --hf-key "your_hf_token" --wandb-key "your_wandb_key"
   ```

4. **Run inference**:
   ```bash
   uv run app/inferencer.py --hf-key "your_hf_token"
   ```

5. **Evaluate results**:
   ```bash
   uv run app/evaluator.py --openai-key "your_openai_key"
   ```

## 🏗️ Architecture

The pipeline consists of three main components:

- **Fine-Tuner** (`app/finetuner.py`): LoRA-based model fine-tuning
- **Inferencer** (`app/inferencer.py`): Model inference and prediction
- **Evaluator** (`app/evaluator.py`): Comprehensive model evaluation

## 📊 Supported Models

- Qwen 2.5 series
- Llama models  
- Mistral models
- Any model compatible with Unsloth

## 🛠️ Development

### Building Documentation

```bash
# Install docs dependencies
uv sync --extra docs

# Serve locally
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

### Running Tests

```bash
uv run pytest
```

## 🤝 Contributing

Please read our [Contributing Guide](https://your-username.github.io/Fine-Tune-Pipeline/contributing/) for details on how to contribute to this project.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [Hugging Face](https://huggingface.co/) for models and datasets
- [Weights & Biases](https://wandb.ai/) for experiment tracking
