import unsloth  # type: ignore # Do not remove this import, it is required for the unsloth library to work properly
from unsloth import FastLanguageModel

import argparse
import json
import os
from tqdm import tqdm

# Local imports
from config_manager import get_config_manager, InferencerConfig
from utils import (
    load_huggingface_dataset,
    push_dataset_to_huggingface,
    setup_run_name,
    login_huggingface,
)


class Inferencer:
    def __init__(self, *, config_manager=get_config_manager()):
        self.config = InferencerConfig.from_config(config_manager)

        # TODO: Load the model name from the previous finetuner step trained model iD
        self.model = None
        self.tokenizer = None

        # NOTE: IF CHANGED, UPDATE THE **FINETUNER** AS WELL
        self.MODEL_LOCAL_INPUT_DIR = "./models/fine_tuned"

        # NOTE: IF CHANGED, UPDATE THE **EVALUATOR** AS WELL
        self.OUTPUT_SYSTEM_PROMPT_COLUMN = "system_prompt"
        self.OUTPUT_USER_PROMPT_COLUMN = "user_prompt"
        self.OUTPUT_ASSISTANT_RESPONSE_COLUMN = "assistant_response"
        self.OUTPUT_GROUND_TRUTH_COLUMN = "ground_truth"
        self.OUTPUT_FILE_NAME = "inferencer_output.jsonl"

    def get_system_prompt(self, data_row):
        """
        Get the system prompt from the configuration or dataset.
        """
        if self.config.system_prompt_override_text is not None:
            system_part = self.config.system_prompt_override_text.strip()
        elif self.config.system_prompt_column is not None:
            system_part = data_row[self.config.system_prompt_column].strip()
        else:
            system_part = None
        return system_part

    def get_user_question(self, data_row):
        """
        Get the question column from the dataset.
        """
        if self.config.question_column is None:
            raise ValueError("Question column is not specified in the configuration.")
        question = data_row[self.config.question_column].strip()
        return question

    def get_ground_truth(self, data_row):
        """
        Get the ground truth from the dataset.
        """
        if self.config.ground_truth_column is None:
            raise ValueError(
                "Ground truth column is not specified in the configuration."
            )
        ground_truth = data_row[self.config.ground_truth_column].strip()
        return ground_truth

    # TODO: Can we do this as a batch like we do in training?
    def convert_a_data_row_to_conversation_format(self, data_row):
        """
        Convert a single data_row to a conversation format.
        """
        system_part = self.get_system_prompt(data_row)
        user_part = data_row[self.config.question_column].strip()
        conversation = [{"role": "user", "content": user_part}]
        if system_part is not None:
            conversation.insert(0, {"role": "system", "content": system_part})
        return conversation

    def apply_chat_template_to_conversation(self, conversation):
        """Apply the chat template to a conversation.
        This method tokenizes the conversation and prepares it for model input.
        """
        if self.tokenizer is None or self.model is None:
            raise ValueError(
                "Model or tokenizer is not initialized. Please load the model and tokenizer first."
            )
        tokenized_msg = self.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(self.model.device)
        return tokenized_msg

    def generate_a_response(self, data_row):
        """
        Generate a response for a single data_row.
        This method converts the data_row to conversation format, tokenizes it,
        and generates a response using the model.

        Args:
            data_row (dict): A single row of data containing user prompt and optional system prompt.
        Returns:
            dict: A dictionary containing the system prompt, user prompt, and model response.
        Example:
            data_row = {
                "system_prompt": "You are a helpful assistant.",
                "question": "What is the capital of France?",
                "assistant": "Paris"
            }
            response = inferencer.generate_a_response(data_row)
            print(response)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError(
                "Model or tokenizer is not initialized. Please load the model and tokenizer first."
            )

        conversation = self.convert_a_data_row_to_conversation_format(data_row)
        tokenized_msg = self.apply_chat_template_to_conversation(conversation)

        output_ids = self.model.generate(
            input_ids=tokenized_msg,
            max_new_tokens=self.config.max_new_tokens,
            use_cache=self.config.use_cache,
            temperature=self.config.temperature,
            min_p=self.config.min_p,
        )
        model_response = self.tokenizer.decode(
            output_ids[0][tokenized_msg.shape[-1] :], skip_special_tokens=True
        )
        # output_ids[0][tokenized_msg.shape[-1]:] --> output_ids[0] is the generated response, and we skip the user and system parts by slicing from the end of the tokenized message.
        output_data_row = {
            self.OUTPUT_SYSTEM_PROMPT_COLUMN: self.get_system_prompt(data_row),
            self.OUTPUT_USER_PROMPT_COLUMN: self.get_user_question(data_row),
            self.OUTPUT_ASSISTANT_RESPONSE_COLUMN: model_response,
            self.OUTPUT_GROUND_TRUTH_COLUMN: self.get_ground_truth(data_row),
        }
        return output_data_row

    def save_datarow_to_jsonl(self, file_name, data_row):
        """
        Save a single data_row to a JSONL file. If the file does not exist, it creates the file.
        If the file exists, it appends the data_row to the file.

        Args:
            file_name (str): The name of the JSONL file.
            data_row (dict): The data row to save.
        """
        mode = "a" if os.path.exists(file_name) else "w"
        with open(file_name, mode, encoding="utf-8") as f:
            f.write(json.dumps(data_row, ensure_ascii=False) + "\n")

    def run(self):
        """
        Generate responses for each user prompt in the dataset and return as a list of dicts.
        """
        # Login to HF
        login_huggingface()
        print("--- ✅ Login to Hugging Face Hub successful. ---")

        # Load the dataset
        testing_dataset = load_huggingface_dataset(self.config.testing_data_id)
        print("--- ✅ Loaded testing dataset successfully. ---")

        # Load the model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.MODEL_LOCAL_INPUT_DIR,
            max_seq_length=self.config.max_sequence_length,
            dtype=self.config.dtype,
            load_in_4bit=self.config.load_in_4bit,
            load_in_8bit=self.config.load_in_8bit,
            token=os.getenv("HF_TOKEN"),
        )
        print("--- ✅ Loaded model and tokenizer successfully. ---")
        FastLanguageModel.for_inference(self.model)
        print("--- ✅ Model set for inference. ---")

        self.run_name = setup_run_name(
            name=self.config.run_name,
            prefix=self.config.run_name_prefix,
            suffix=self.config.run_name_suffix,
        )
        print(f"--- ✅ Run name set to: {self.run_name} ---")

        # Model response generation
        print("--- ✅ Starting inference on the testing dataset. ---")
        for data_row in tqdm(
            testing_dataset, desc="Generating responses", unit="data_row"
        ):
            response = self.generate_a_response(data_row)
            self.save_datarow_to_jsonl(self.OUTPUT_FILE_NAME, response)
            push_dataset_to_huggingface(
                repo_id=f"{self.config.hf_user_id}/{self.run_name}",
                dataset_path=self.OUTPUT_FILE_NAME,
            )
        print(
            f"--- ✅ Responses saved to {self.OUTPUT_FILE_NAME} and pushed to HuggingFace Hub under {self.config.hf_user_id}/{self.run_name} ---"
        )
        print("--- ✅ Inference completed successfully. ---")


if __name__ == "__main__":
    print("--- Starting the Inferencer ---")
    parser = argparse.ArgumentParser(description="Inference the language model")
    parser.add_argument("--hf-key", type=str, help="Hugging Face API token")

    args = parser.parse_args()
    
    # Set environment variables from command-line arguments
    if args.hf_key:
        os.environ["HF_TOKEN"] = args.hf_key
    print(f"--- Argparser done ---")
    inferencer = Inferencer()
    print(f"--- Inferencer initialized with config: {inferencer.config} ---")
    print(f"--- Starting the inference run ---")
    inferencer.run()
