# Installation

This guide will help you set up the Fine-Tune Pipeline on your system.

## Prerequisites

Before installing the Fine-Tune Pipeline, ensure you have:

- **Python 3.12 or higher**
- **uv** package manager (recommended) or pip
- **Git** for version control
- **CUDA-compatible GPU** (required for fine tune and inference phases)

## Installing uv (Recommended)

We recommend using `uv` as the package manager for this project as it provides faster dependency resolution and better environment management.

### Windows

```bash
# Using PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### macOS and Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Alternative: Using pip

```bash
pip install uv
```

## Clone the Repository

```bash
git clone https://github.com/Fine-Tuning-Team/Fine-Tune-Pipeline.git
cd Fine-Tune-Pipeline
```

## Install Dependencies

```bash
# Sync all dependencies (this will create a virtual environment automatically)
uv sync
```

## Verify Installation

To verify that everything is installed correctly, run:

```bash
# Check if the main module can be imported
uv run python -c "from app.finetuner import FineTune; print('✅ Installation successful!')"
```

## GPU Setup (Optional but Recommended)

For optimal performance, especially during fine-tune and inference phases, a CUDA-compatible GPU is required.

### CUDA Installation

1. **Check CUDA compatibility**:

   ```bash
   nvidia-smi
   ```

2. **Install PyTorch with CUDA support**:

   ```bash
   # For CUDA 11.8
   uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify CUDA installation**:

   ```bash
   uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Next Steps

Now that you have the pipeline installed, you can:

1. [Set up your environment](environment-setup.md) with API keys and configurations
2. Follow the [Quick Start Guide](quick-start.md) to run your first fine-tuning job
3. Explore the [Configuration Options](../configuration/overview.md) to customize your setup

## Troubleshooting

### Common Issues

**Issue**: `uv sync` fails with dependency conflicts

```bash
# Solution: Update uv and try again
uv self update
uv sync --refresh
```

**Issue**: CUDA out of memory during training

```bash
# Solution: Reduce batch size in config.toml
device_train_batch_size = 2  # Reduce from default 4
device_validation_batch_size = 2
```

**Issue**: Import errors with unsloth

```bash
# Solution: Reinstall unsloth
uv remove unsloth
uv add "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"
```

For more troubleshooting tips, see our [Troubleshooting Guide](../troubleshooting.md).
