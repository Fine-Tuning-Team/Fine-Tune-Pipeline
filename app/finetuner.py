import unsloth  # type: ignore
from unsloth import FastLanguageModel, is_bfloat16_supported # type: ignore
from unsloth.chat_templates import get_chat_template, train_on_responses_only # type: ignore

# import torch
import datasets
import os
import sys
import wandb
import pandas as pd
import time

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from huggingface_hub import login
from trl import SFTTrainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


from app.config_manager import get_config_manager, FineTunerConfig


class FineTune:
    def __init__(self, *, config_manager=get_config_manager(), FineTunerConfig=FineTunerConfig):
        # Load configuration
        self.config = FineTunerConfig.from_config(config_manager)
        
        # Constants
        self.CONVERSATIONS_KEY = "conversations"  # Key for conversations when transforming dataset (refer to _convert_to_conversations method)
        self.TEXTS_KEY = "text"  # Key for text when formatting prompts (refer to _formatting_prompts_func method)

        # Instance variables
        self.model = None
        self.tokenizer = None
        self.run_name = None   


    def load_base_model_and_tokenizer(self) -> tuple[FastLanguageModel, PreTrainedTokenizerBase]:
        """
        Load the base model and tokenizer from the specified model name.
        Returns:
            FastLanguageModel: The loaded model.
            Tokenizer: The tokenizer associated with the model.
        """
        return FastLanguageModel.from_pretrained(
            model_name = self.config.base_model_id,
            max_seq_length = self.config.max_sequence_length,
            dtype = self.config.dtype,
            load_in_4bit = self.config.load_in_4bit,
            load_in_8bit = self.config.load_in_8bit,
            full_finetuning = self.config.full_finetuning,
            token = os.getenv("HF_TOKEN")   # For gated models
        )

    def get_peft_model(self) -> FastLanguageModel:
        """
        Convert the loaded model to a PEFT (Parameter-Efficient Fine-Tuning) model.
        Returns:
            FastLanguageModel: The PEFT model.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before converting to PEFT model.")
        return FastLanguageModel.get_peft_model(
            self.model,
            r = self.config.rank,
            target_modules = self.config.target_modules,
            lora_alpha = self.config.lora_alpha,
            lora_dropout = self.config.lora_dropout,
            bias = self.config.bias,
            use_gradient_checkpointing = self.config.use_gradient_checkpointing,
            random_state = self.config.seed,
            use_rslora = self.config.use_rslora,
            loftq_config = self.config.loftq_config,
        )

    def convert_to_conversations(self, data_row) -> dict:
        """
        Convert a single data_row to a conversation format.
        Args:
            data_row (dict): A single data_row from the dataset.
        Returns:
            dict: A dictionary containing the conversation format (System prompt is optional).
        Example:
            {
                "conversations": [
                    {"role": "system", "content": "System prompt here"},
                    {"role": "user", "content": "User question here"},
                    {"role": "assistant", "content": "Assistant answer here"}
                ]
            }
        """
        if self.config.system_prompt_override_text is not None:
            system_part = self.config.system_prompt_override_text.strip()
        elif self.config.system_prompt_column is not None:
            system_part = data_row[self.config.system_prompt_column].strip()
        else:
            system_part = None
        user_part = data_row[self.config.question_column].strip()
        ground_truth = data_row[self.config.ground_truth_column].strip()
        output = {
            self.CONVERSATIONS_KEY: [
                {"role": "user", "content": user_part},
                {"role": "assistant", "content": ground_truth}
            ]
        }
        if system_part is not None:
            output[self.CONVERSATIONS_KEY].insert(0, {"role": "system", "content": system_part})
        return output


    def formatting_prompts_func(self, data_rows):
        """
        Format the conversations for training by applying the chat template.
        Args:
            data_rows (dict): A batch of data_rows from the dataset.
        Returns:
            dict: A dictionary containing the formatted text for training.
        Example:
            {
                "text": [
                    "<|im_start|>system\nSystem prompt here<|im_end|>\n<|im_start|>user\nUser question here<|im_end|>\n<|im_start|>assistant\n",
                    "<|im_start|>system\nSystem prompt here<|im_end|>\n<|im_start|>user\nUser question here<|im_end|>\n<|im_start|>assistant\n",
                    ...
                ]
            }
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be loaded before formatting prompts.")
        if not hasattr(self.tokenizer, 'apply_chat_template'):
            # TODO: What to do if tokenizer does not have apply_chat_template method?
            raise NotImplementedError("Tokenizer does not have 'apply_chat_template' method. Adding a template manually is not supported yet.")
        convos = data_rows[self.CONVERSATIONS_KEY]
        texts = [self.tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        # NOTE:No need for generation prompt when training
        return { self.TEXTS_KEY : texts, }
    
    def setup_run_name(self):
        """
        Set up the run name for the training process.
        The run name is constructed from the base model ID, project name, and optional prefixes/suffixes.
        """
        if self.config.run_name is None:
            self.run_name = str(int(time.time()))   # in seconds since epoch
        self.run_name = self.config.run_name_prefix + self.run_name + self.config.run_name_suffix # type: ignore


    def handle_wandb_setup(self):
        """
        Handle the setup for Weights & Biases (wandb) logging.
        Returns:
            str: The run name used for the Weights & Biases run.
        """
        wandb.login(key=os.getenv('WANDB_TOKEN'))
        
        wandb_run_name = self.run_name
        wandb.init(project=self.config.wandb_project_name, name=wandb_run_name)


    def login_huggingface(self):
        """
        Log in to Hugging Face using the token from environment variables.
        Raises:
            ValueError: If the Hugging Face token is not set in the environment variables.
        """
        login(token=os.getenv("HF_TOKEN"))


    def load_huggingface_dataset(self, dataset_id: str) -> (DatasetDict | Dataset | IterableDatasetDict | IterableDataset):
        """
        Load a dataset from Hugging Face. Whether the data is jsonl, csv, parquet or any other format, it will be loaded as a Hugging Face Dataset.
        Args:
            dataset_id (str): The ID of the dataset on Hugging Face.
        Returns:
            datasets.arrow_dataset.Dataset: The loaded dataset.
        """
        try:
            dataset = datasets.load_dataset(dataset_id)
            if isinstance(dataset, dict):
                # If the dataset is a dictionary, return the first split
                return list(dataset.values())[0]
            return dataset
        except Exception as e:
            raise ValueError(f"Failed to load dataset from Hugging Face with ID '{dataset_id}': {e}")

    def get_columns_to_remove(self, dataset: Dataset | DatasetDict | IterableDatasetDict | IterableDataset, dataset_id: str) -> list[str]:
        """
        Get the columns to remove from the dataset based on the configuration.
        Args:
            dataset_columns (list[str]): The list of columns in the dataset.
        Returns:
            list[str]: The list of columns to remove.
        """
        if isinstance(dataset, IterableDatasetDict):
            raise NotImplementedError(f"Cannot determine columns for IterableDatasetDict of {dataset_id}. Iterable datasets are not supported yet.")
        dataset_columns = dataset.column_names
        if dataset_columns is None:
            raise ValueError(f"Dataset {dataset_id} does not have column names. Ensure the dataset is properly loaded.")
        
        columns_to_keep = {self.config.question_column, self.config.ground_truth_column}
        if (self.config.system_prompt_column is not None) and (self.config.system_prompt_override_text is None):
            columns_to_keep.add(self.config.system_prompt_column)
        
        return [col for col in dataset_columns if col not in columns_to_keep]

    def run(self):
        """
        Run the fine-tuning process.
        Returns:
            TrainerStats: The statistics from the training process.
        """
        # Load training and validation data
        training_dataset = self.load_huggingface_dataset(self.config.training_data_id)
        if self.config.validation_data_id is not None:
            validation_dataset = self.load_huggingface_dataset(self.config.validation_data_id)
        else:
            validation_dataset = None

        # Initialize model and tokenizer
        self.model, self.tokenizer = self.load_base_model_and_tokenizer()
        self.model = self.get_peft_model()

        # Data operations
        # strip doesnt work with batched=True, so we use batched=False
        training_dataset = training_dataset.map(
            self.convert_to_conversations, 
            remove_columns=self.get_columns_to_remove(training_dataset, self.config.training_data_id), 
            batched=False
        )   
        training_dataset = training_dataset.map(self.formatting_prompts_func, batched=True)
        if validation_dataset is not None and self.config.validation_data_id is not None:
            validation_dataset = validation_dataset.map(
                self.convert_to_conversations, 
                remove_columns=self.get_columns_to_remove(validation_dataset, self.config.validation_data_id), 
                batched=False
            )   
            validation_dataset = validation_dataset.map(self.formatting_prompts_func, batched=True)

        self.handle_wandb_setup()

        # Training
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer, # type: ignore
            train_dataset = training_dataset,
            eval_dataset = validation_dataset,
            dataset_text_field = self.TEXTS_KEY,    # type: ignore
            max_seq_length = self.config.max_sequence_length,    # type: ignore
            data_collator = DataCollatorForSeq2Seq(tokenizer = self.tokenizer),
            dataset_num_proc = self.config.dataset_num_proc,     # type: ignore
            packing = self.config.packing,   # type: ignore
            args = TrainingArguments(
                per_device_train_batch_size = self.config.device_train_batch_size,
                per_device_eval_batch_size= self.config.device_validation_batch_size,
                gradient_accumulation_steps = self.config.grad_accumulation,
                warmup_steps = self.config.warmup_steps,
                num_train_epochs = self.config.epochs,
                learning_rate = self.config.learning_rate,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = self.config.log_steps,
                logging_first_step=self.config.log_first_step,
                optim = self.config.optimizer,
                weight_decay = self.config.weight_decay,
                lr_scheduler_type = self.config.lr_scheduler_type,
                seed = self.config.seed,
                output_dir = self.config.model_local_output_dir,  # Save checkpoints and outputs to local models dir
                report_to = self.config.report_to,
                save_steps=self.config.save_steps,
                save_total_limit=self.config.save_total_limit,
                push_to_hub=self.config.push_to_hub,
                hub_model_id=self.run_name
            ),
            callbacks = [None]
        )

        if self.config.train_on_responses_only:
            trainer = train_on_responses_only(
                trainer,
                instruction_part = self.config.question_part,
                response_part = self.config.answer_part,
            )

        trainer_stats = trainer.train()
        return trainer_stats

if __name__ == "__main__":
    print("[ERROR] Please run the pipeline using main.py, not directly")
    sys.exit(1)

