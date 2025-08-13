# Contributing

We welcome contributions to the Fine-Tune Pipeline! This document outlines how to contribute to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/Fine-Tune-Pipeline.git
   cd Fine-Tune-Pipeline
   ```
3. **Set up development environment**:
   ```bash
   uv sync
   ```

## Development Workflow

### 1. Create a Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Test Your Changes

```bash
# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_finetuner.py

# Run with coverage
uv run pytest --cov=app
```

### 4. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with descriptive message
git commit -m "feat: add new feature description"

# Or for bug fixes
git commit -m "fix: resolve issue with specific problem"
```

### 5. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Then create a Pull Request on GitHub
```

## Code Style

### Python Style Guide

- Follow PEP 8
- Use Black for code formatting
- Use type hints where appropriate
- Write docstrings for all functions and classes

### Formatting

```bash
# Format code with Black
uv run black app/ tests/

# Check formatting
uv run black --check app/ tests/
```

### Type Checking

```bash
# Install mypy
uv add --dev mypy

# Run type checking
uv run mypy app/
```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test
uv run pytest tests/test_finetuner.py::test_model_loading
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies when appropriate

Example test:

```python
import pytest
from app.finetuner import FineTune
from app.config_manager import ConfigManager

def test_finetuner_initialization():
    """Test that FineTune initializes correctly with valid config."""
    config_manager = ConfigManager("config.toml")
    tuner = FineTune(config_manager=config_manager)
    
    assert tuner.config is not None
    assert tuner.model is None  # Not loaded yet
    assert tuner.tokenizer is None  # Not loaded yet

def test_invalid_config_raises_error():
    """Test that invalid configuration raises appropriate error."""
    with pytest.raises(FileNotFoundError):
        ConfigManager("nonexistent_config.toml")
```

## Documentation

### Building Documentation Locally

```bash
# Install docs dependencies
uv sync --extra docs

# Serve documentation locally
uv run mkdocs serve

# Build documentation
uv run mkdocs build
```

### Writing Documentation

- Use clear, concise language
- Include code examples
- Add screenshots for UI elements
- Update table of contents as needed

### Documentation Structure

```
docs/
├── index.md                  # Home page
├── getting-started/          # Setup and quick start
│   ├── environment-setup.md
│   └── quick-start.md
├── configuration/            # Configuration guides
│   ├── overview.md
│   ├── fine-tuner.md
│   ├── inferencer.md
│   └── evaluator.md
├── components/               # Component documentation
│   ├── fine-tuner.md
│   ├── inferencer.md
│   └── evaluator.md
├── tutorials/                # Step-by-step tutorials
│   ├── basic-fine-tuning.md
│   ├── advanced-configuration.md
│   └── ci-cd-integration.md
├── api-reference.md          # API documentation
├── troubleshooting.md        # Common issues and solutions
└── contributing.md           # This file
```

## Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH` (e.g., `1.2.3`)
- Major: Breaking changes
- Minor: New features, backward compatible
- Patch: Bug fixes, backward compatible

### Creating a Release

1. **Update version** in `pyproject.toml`
2. **Update changelog** with new features and fixes
3. **Create release branch**:
   ```bash
   git checkout -b release/v1.2.3
   ```
4. **Run tests** and ensure everything works
5. **Create pull request** for the release
6. **Tag the release** after merging:
   ```bash
   git tag v1.2.3
   git push origin v1.2.3
   ```

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

- **Clear description** of the issue
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, GPU)
- **Configuration file** (remove sensitive information)
- **Error messages** and stack traces

### Feature Requests

When requesting features, please include:

- **Clear description** of the desired functionality
- **Use case** or motivation for the feature
- **Proposed implementation** (if you have ideas)
- **Examples** of how it would be used

## Code of Conduct

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards other community members

### Enforcement

Project maintainers are responsible for clarifying standards and will take appropriate action in response to any behavior that violates this code of conduct.

## Development Setup

### Prerequisites

- Python 3.12 or higher
- Git
- uv package manager
- CUDA-compatible GPU (optional but recommended)

### Environment Setup

```bash
# Clone repository
git clone https://github.com/your-username/Fine-Tune-Pipeline.git
cd Fine-Tune-Pipeline

# Install dependencies
uv sync

# Install development dependencies
uv sync --extra docs

# Set up pre-commit hooks (optional)
uv add --dev pre-commit
uv run pre-commit install
```

### IDE Configuration

#### VS Code

Recommended extensions:
- Python
- Pylance
- Black Formatter
- autoDocstring
- GitLens

Settings (`.vscode/settings.json`):
```json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

#### PyCharm

- Enable Black formatter
- Set up pytest as test runner
- Configure type checking with mypy

## Contribution Guidelines

### Pull Request Process

1. **Ensure tests pass** before submitting
2. **Update documentation** for new features
3. **Follow commit message conventions**
4. **Keep changes focused** - one feature per PR
5. **Respond to feedback** promptly

### Commit Message Format

```
<type>(<scope>): <description>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(finetuner): add support for custom chat templates

fix(config): resolve TOML parsing error for null values

docs(tutorial): add advanced configuration examples
```

### Review Process

All submissions require review from maintainers:

1. **Automated checks** must pass (tests, linting)
2. **Code review** by at least one maintainer
3. **Documentation review** for user-facing changes
4. **Performance review** for core functionality changes

## Getting Help

If you need help contributing:

1. **Check existing issues** for similar questions
2. **Read the documentation** thoroughly
3. **Join our community** discussions
4. **Ask questions** in issues or discussions
5. **Reach out to maintainers** directly if needed

## Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **GitHub contributors** page
- **Documentation acknowledgments**

Thank you for contributing to the Fine-Tune Pipeline! 🚀
