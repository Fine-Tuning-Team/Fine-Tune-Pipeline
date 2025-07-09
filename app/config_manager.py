import os
import toml

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """Centralized configuration manager for the pipeline."""

    def __init__(self, config_path: str = "config.toml"):
        self.config_path = Path(config_path)
        self._config = None
        self.load_config()

    def load_config(self) -> None:
        """Load configuration from TOML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            self._config = toml.load(f)

        # Replace environment variable placeholders
        self._resolve_env_vars(self._config)

        # Convert string "null" values to Python None
        self._convert_null_strings(self._config)

    def _convert_null_strings(self, config: Dict[str, Any]) -> None:
        """Recursively convert string 'null' values to Python None."""
        for key, value in config.items():
            if isinstance(value, dict):
                self._convert_null_strings(value)
            elif isinstance(value, list):
                # Handle lists that might contain "null" strings
                config[key] = [None if item == "null" else item for item in value]
            elif value == "null":
                config[key] = None

    def _resolve_env_vars(self, config: Dict[str, Any]) -> None:
        """Recursively resolve environment variables in config values."""
        for key, value in config.items():
            if isinstance(value, dict):
                self._resolve_env_vars(value)
            elif (
                isinstance(value, str)
                and value.startswith("${")
                and value.endswith("}")
            ):
                env_var = value[2:-1]
                config[key] = os.getenv(env_var, value)

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a specific configuration section."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded")

        if section not in self._config:
            raise KeyError(f"Configuration section '{section}' not found")

        return self._config[section].copy()  # to avoid direct modification

    def get_value(self, section: str, key: str, default: Any = None) -> Any:
        """Get a specific configuration value."""
        section_config = self.get_section(section)
        return section_config.get(key, default)

    def validate_section(self, section: str, required_keys: list) -> None:
        """Validate that a section contains all required keys."""
        section_config = self.get_section(section)
        missing_keys = [key for key in required_keys if key not in section_config]

        if missing_keys:
            raise ValueError(
                f"Missing required configuration keys in '{section}': {missing_keys}"
            )

    def validate_dataclass_config(self, section: str, dataclass_type) -> None:
        """Validate that a section contains all fields required by a dataclass."""
        from dataclasses import fields

        section_config = self.get_section(section)
        required_fields = [field.name for field in fields(dataclass_type)]
        missing_keys = [
            field for field in required_fields if field not in section_config
        ]

        if missing_keys:
            raise ValueError(
                f"Missing required configuration keys in '{section}' for {dataclass_type.__name__}: {missing_keys}"
            )

        # Optional: Check for extra keys that don't belong to the dataclass
        extra_keys = [
            key for key in section_config.keys() if key not in required_fields
        ]
        if extra_keys:
            print(
                f"Warning: Extra configuration keys in '{section}' not used by {dataclass_type.__name__}: {extra_keys}"
            )


# Dataclass approach for type safety
@dataclass
class FineTunerConfig:
    base_model_id: str  # Hugging Face model ID or local path
    # TODO: Do we need a description? Like a model description? or run description?
    # fine_tuned_model_id: str -> Let's create a random id from mlflow and push
    is_multimodel: bool  # True if the model is multimodal (vision + language)
    finetune_vision_layers: bool  # Whether to finetune vision layers
    finetune_language_layers: bool  # Whether to finetune language layers
    finetune_attention_modules: bool  # Whether to finetune attention modules
    finetune_mlp_modules: bool  # Whether to finetune MLP modules
    max_sequence_length: int
    dtype: int | None  # Can be null
    load_in_4bit: bool
    load_in_8bit: bool
    full_finetuning: bool
    rank: int
    lora_alpha: int
    lora_dropout: int
    target_modules: list
    bias: str
    training_data_id: str  # Hugging Face dataset ID or local path
    validation_data_id: str | None  # Optional, can be None
    dataset_num_proc: int  # Number of processes for dataset loading
    question_column: str
    ground_truth_column: str
    system_prompt_column: str | None  # Optional, can be None
    system_prompt_override_text: str | None  # Optional, can be None
    run_name: str | None  # Leave empty for random name
    run_name_prefix: str
    run_name_suffix: str
    wandb_project_name: str
    device_train_batch_size: int
    device_validation_batch_size: int
    grad_accumulation: int
    epochs: int
    learning_rate: float
    warmup_steps: int
    optimizer: str  # TODO: Need constraints here
    weight_decay: float
    lr_scheduler_type: str  # Need constraints here
    seed: int  # Random seed for reproducibility
    log_steps: int  # Log every n steps
    log_first_step: bool
    save_steps: int  # Save every n steps
    save_total_limit: int  # Limit the number of saved checkpoints
    push_to_hub: bool  # Push the model to Hugging Face Hub
    report_to: str  # Reporting tool, e.g., "wandb", "tensorboard", "none"
    packing: bool  # Can make 5x training faster, for shorter sequences
    use_gradient_checkpointing: (
        str | bool
    )  # Can be True, False, or 'unsloth' - for very large contexts
    use_flash_attention: bool
    use_rslora: bool
    loftq_config: Any  # Can be null
    question_part: str
    answer_part: str
    train_on_responses_only: bool  # If True, only train (i.e., calculate loss) on responses, not questions. Refer: https://github.com/unslothai/unsloth/issues/823

    @classmethod
    def from_config(cls, config_manager: ConfigManager):
        config_manager.validate_dataclass_config("fine_tuner", cls)
        section = config_manager.get_section("fine_tuner")
        return cls(**section)


@dataclass
class InferencerConfig:
    max_sequence_length: int
    dtype: Any
    load_in_4bit: bool
    load_in_8bit: bool
    testing_data_id: str
    question_column: str
    ground_truth_column: str
    system_prompt_column: str | None  # Optional, can be None
    system_prompt_override_text: str | None  # Optional, can be None
    max_new_tokens: int
    use_cache: bool
    temperature: float
    min_p: float
    hf_user_id: str  # TODO: Should this be a secret? This will be used to generate the model path
    run_name: str | None  # Leave empty for random name
    run_name_prefix: str
    run_name_suffix: str

    @classmethod
    def from_config(cls, config_manager: ConfigManager):
        config_manager.validate_dataclass_config("inferencer", cls)
        section = config_manager.get_section("inferencer")
        return cls(**section)


@dataclass
class EvaluatorConfig:
    metrics: list[str]
    llm: str
    embedding: str
    run_name: str | None
    run_name_prefix: str
    run_name_suffix: str
    cost_per_million_input_tokens: float
    cost_per_million_output_tokens: float
    hf_user_id: str  # Hugging Face user ID for repository creation

    @classmethod
    def from_config(cls, config_manager: ConfigManager):
        config_manager.validate_dataclass_config("evaluator", cls)
        section = config_manager.get_section("evaluator")
        return cls(**section)


@dataclass
class MLFlowConfig:
    tracking_uri: str
    experiment_name: str
    run_name: str | None
    run_name_prefix: str
    run_name_suffix: str

    @classmethod
    def from_config(cls, config_manager: ConfigManager):
        config_manager.validate_dataclass_config("mlflow", cls)
        section = config_manager.get_section("mlflow")
        return cls(**section)


# Global config instance (singleton pattern)
_config_manager = None


def get_config_manager(config_path: str = "config.toml") -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager
