import os
import toml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass


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
        
        with open(self.config_path, 'r') as f:
            self._config = toml.load(f)
        
        # Replace environment variable placeholders
        self._resolve_env_vars(self._config)
    
    def _resolve_env_vars(self, config: Dict[str, Any]) -> None:
        """Recursively resolve environment variables in config values."""
        for key, value in config.items():
            if isinstance(value, dict):
                self._resolve_env_vars(value)
            elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                config[key] = os.getenv(env_var, value)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a specific configuration section."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        
        if section not in self._config:
            raise KeyError(f"Configuration section '{section}' not found")
        
        return self._config[section].copy() # to avoid direct modification
    
    def get_value(self, section: str, key: str, default: Any = None) -> Any:
        """Get a specific configuration value."""
        section_config = self.get_section(section)
        return section_config.get(key, default)
    
    def validate_section(self, section: str, required_keys: list) -> None:
        """Validate that a section contains all required keys."""
        section_config = self.get_section(section)
        missing_keys = [key for key in required_keys if key not in section_config]
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys in '{section}': {missing_keys}")


# Dataclass approach for type safety
@dataclass
class FineTunerConfig:
    base_model_id: str  # Hugging Face model ID or local path
    # TODO: Do we need a description? Like a model description? or run description?
    # fine_tuned_model_id: str -> Let's create a random id from mlflow and push
    max_sequence_length: int
    dtype: Any  # Can be null
    load_in_4bit: bool
    load_in_8bit: bool
    full_finetuning: bool
    rank: int
    lora_alpha: int
    lora_dropout: int
    target_modules: list
    bias: str
    training_data_id: str # Hugging Face dataset ID or local path
    dataset_num_proc: int # Number of processes for dataset loading
    question_column: str  
    ground_truth_column: str  
    system_prompt_column: str | None  # Optional, can be None
    system_prompt_override_text: str | None  # Optional, can be None
    wandb_run_name: str | None # Leave empty for random name
    wandb_run_name_prefix: str | None # Leave empty for no prefix
    wandb_run_name_suffix: str | None # Leave empty for no suffix
    wandb_project_name: str
    device_batch_size: int
    grad_accumulation: int
    epochs: int
    learning_rate: float
    warmup_steps: int
    optimizer: str  # TODO: Need constraints here
    weight_decay: float
    lr_scheduler_type: str # Need constraints here
    seed: int   # Random seed for reproducibility
    log_steps: int # Log every n steps
    log_first_step: bool
    save_steps: int # Save every n steps
    save_total_limit: int   # Limit the number of saved checkpoints
    push_to_hub: bool       # Push the model to Hugging Face Hub
    packing: bool   # Can make 5x training faster, for shorter sequences
    use_gradient_checkpointing: str | bool # Can be True, False, or 'unsloth' - for very large contexts
    use_flash_attention: bool
    use_rslora: bool
    loftq_config: Any  # Can be null
    question_part: str
    answer_part: str


    @classmethod
    def from_config(cls, config_manager: ConfigManager):
        section = config_manager.get_section("fine_tuner")
        return cls(**section)


@dataclass
class InferencerConfig:
    max_sequence_length: int
    dtype: Any
    load_in_4bit: bool
    load_in_8bit: bool
    training_data_id: str
    question_column: str
    ground_truth_column: str
    system_prompt_column: str | None  # Optional, can be None
    system_prompt_override_text: str | None  # Optional, can be None
    max_new_tokens: int
    use_cache: bool
    temperature: float
    min_p: float
    hf_user_id: str # TODO: Should this be a secret? This will be used to generate the model path
    
    @classmethod
    def from_config(cls, config_manager: ConfigManager):
        section = config_manager.get_section("inferencer")
        return cls(**section)


@dataclass
class EvaluatorConfig:
    metrics: list[str]
    llm_model_id: str
    embedding_model_id: str | None  # Optional, can be None
    
    @classmethod
    def from_config(cls, config_manager: ConfigManager):
        section = config_manager.get_section("evaluator")
        return cls(**section)
