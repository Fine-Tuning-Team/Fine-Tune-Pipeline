import argparse
import json
import os

from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
)
from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics import (
    SemanticSimilarity,
    AnswerAccuracy,
    FactualCorrectness,
    RougeScore,
    BleuScore,
    AnswerCorrectness,
    AnswerRelevancy,
    Metric,
)
from ragas.cost import get_token_usage_for_openai, TokenUsage
import mlflow
import mlflow.data

# Local imports
from config_manager import get_config_manager, EvaluatorConfig
from utils import log_configurations_to_mlflow, setup_run_name, setup_openai_key, login_huggingface, push_dataset_to_huggingface


class Evaluator:
    def __init__(self, *, config_manager=get_config_manager()):
        # Load and validate the configuration
        self.config = EvaluatorConfig.from_config(config_manager)

        self.llm = None
        self.embeddings = None
        self.input_dataset: Dataset | None = None
        self.metrics: list[Metric] | None = None
        self.evaluation_results: EvaluationResult | None = None

        # NOTE: IF CHANGED, UPDATE THE **INFERENCER** AS WELL
        self.INPUT_FILE_NAME = "inferencer_output.jsonl"
        self.INPUT_SYSTEM_PROMPT_COLUMN = "system_prompt"
        self.INPUT_USER_PROMPT_COLUMN = "user_prompt"
        self.INPUT_ASSISTANT_RESPONSE_COLUMN = "assistant_response"
        self.INPUT_GROUND_TRUTH_COLUMN = "ground_truth"

        self.OUTPUT_FILE_NAME_DETAILED = "evaluator_output_detailed.xlsx"
        self.OUTPUT_FILE_NAME_SUMMARY = "evaluator_output_summary.json"
        self.SUPPORTED_RAGAS_METRICS = {
            "semantic_similarity": SemanticSimilarity,
            "answer_accuracy": AnswerAccuracy,
            "factual_correctness": FactualCorrectness,
            "rouge_score": RougeScore,
            "bleu_score": BleuScore,
            "answer_correctness": AnswerCorrectness,
            "answer_relevancy": AnswerRelevancy,
        }

    def load_inferencer_output(self) -> None:
        """
        Load the inferencer output dataset.
        """
        if not os.path.exists(self.INPUT_FILE_NAME):
            raise FileNotFoundError(
                f"Input file {self.INPUT_FILE_NAME} does not exist."
            )

        dataset = load_dataset("json", data_files=self.INPUT_FILE_NAME, split="train")

        # Validate required columns
        required_columns = [
            self.INPUT_SYSTEM_PROMPT_COLUMN,
            self.INPUT_USER_PROMPT_COLUMN,
            self.INPUT_ASSISTANT_RESPONSE_COLUMN,
            self.INPUT_GROUND_TRUTH_COLUMN,
        ]

        if isinstance(dataset, (DatasetDict, IterableDatasetDict, IterableDataset)):
            raise NotImplementedError(
                "Loading from IterableDataset or DatasetDict is not supported yet."
            )
        for required_column in required_columns:
            if required_column not in dataset.column_names:
                raise ValueError(
                    f"Missing required column: {required_column} in the input dataset."
                )
        self.input_dataset = dataset

    def load_embeddings(self):
        """
        Load the embeddings model based on the configuration.
        """
        if self.config.embedding is None:
            raise ValueError("Embedding model is not specified in the configuration.")

        self.embeddings = embedding_factory(self.config.embedding)
        if self.embeddings is None:
            raise ValueError(
                f"Failed to load embeddings model: {self.config.embedding}"
            )

    def load_llm(self):
        """
        Load the LLM based on the configuration.
        """
        if self.config.llm is None:
            raise ValueError("LLM is not specified in the configuration.")

        self.llm = llm_factory(self.config.llm)
        if self.llm is None:
            raise ValueError(f"Failed to load LLM: {self.config.llm}")

    def set_ragas_metrics(self) -> None:
        """
        Set the list of supported Ragas metrics functions.
        """
        # Check if all metrics in the user given configuration are supported
        if not all(
            metric in self.SUPPORTED_RAGAS_METRICS for metric in self.config.metrics
        ):
            unsupported_metrics = [
                metric
                for metric in self.config.metrics
                if metric not in self.SUPPORTED_RAGAS_METRICS
            ]
            raise ValueError(
                f"Unsupported metrics found in configuration: {unsupported_metrics}"
            )
        # Return the list of Ragas metrics functions (that are `called`) based on the configuration
        self.metrics = [
            self.SUPPORTED_RAGAS_METRICS[metric]() for metric in self.config.metrics
        ]

    def evaluate(self) -> None:
        """
        Evaluate the inferencer output using the specified metrics.
        """
        if self.input_dataset is None:
            raise ValueError(
                "Input dataset is not loaded. Please load the inferencer output first."
            )
        column_map = {
            "question": self.INPUT_USER_PROMPT_COLUMN,
            "answer": self.INPUT_ASSISTANT_RESPONSE_COLUMN,
            "ground_truth": self.INPUT_GROUND_TRUTH_COLUMN,
        }
        results = evaluate(
            self.input_dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings,
            column_map=column_map,
            token_usage_parser=get_token_usage_for_openai,
        )
        self.evaluation_results = results

    def get_summary_results(self) -> dict:
        """
        Get the summary results of the evaluation.
        """
        if self.evaluation_results is None:
            raise ValueError(
                "Evaluation results are not available. Please run the evaluation first."
            )

        # Convert the evaluation results to a dictionary format
        summary = self.evaluation_results._repr_dict
        return summary
    
    def get_token_count_and_cost(self) -> list:
        """
        Get the token count and cost for the evaluation.
        """
        if self.evaluation_results is None:
            raise ValueError(
                "Evaluation results are not available. Please run the evaluation first."
            )

        # Get token usage for OpenAI
        token_usage_result = self.evaluation_results.total_tokens()
        
        # Handle case where token usage is None or empty
        if token_usage_result is None:
            return []
        
        token_usage: list[TokenUsage] = (
            token_usage_result if isinstance(token_usage_result, list) else [token_usage_result]
        )
        
        # Handle case where token_usage is empty
        if not token_usage:
            return []
        
        cost_per_million_input_tokens = self.config.cost_per_million_input_tokens
        cost_per_million_output_tokens = self.config.cost_per_million_output_tokens

        total_token_usage_and_cost = [
            {
                "input_tokens": token_usage_item.input_tokens,
                "output_tokens": token_usage_item.output_tokens,
                "cost": token_usage_item.cost(cost_per_million_input_tokens*10e-6, cost_per_million_output_tokens*10e-6)
            } for token_usage_item in token_usage if token_usage_item is not None
        ]
        return total_token_usage_and_cost

    def save_results(self) -> None:
        """
        Save the evaluation results to an Excel file.
        """
        if self.evaluation_results is None:
            raise ValueError(
                "Evaluation results are not available. Please run the evaluation first."
            )

        # Save summary results to a JSON file
        if not self.OUTPUT_FILE_NAME_SUMMARY.endswith(".json"):
            raise NotImplementedError(
                "Only JSON format is supported for summary results."
            )
        summary = self.evaluation_results._repr_dict
        with open(self.OUTPUT_FILE_NAME_SUMMARY, "w") as summary_file:
            json.dump(summary, summary_file, indent=4)

        # Save detailed results to an Excel file
        if self.OUTPUT_FILE_NAME_DETAILED.endswith(".xlsx"):
            detailed_df = self.evaluation_results.to_pandas()
            output_file_path = os.path.join("", self.OUTPUT_FILE_NAME_DETAILED)
            detailed_df.to_excel(output_file_path, index=True)

    def push_results_to_huggingface(self) -> None:
        """
        Push the evaluation results to HuggingFace Hub.
        """
        
        if not self.run_name:
            raise ValueError("Run name is not set.")
        
        if self.evaluation_results is None:
            raise ValueError(
                "Evaluation results are not available. Please run the evaluation first."
            )

        # Determine repository names
        summary_repo_name = f"{self.run_name}_summary"
        detailed_repo_name = f"{self.run_name}_detailed"
        
        # Create full repository IDs
        summary_repo_id = f"{self.config.hf_user_id}/{summary_repo_name}"
        detailed_repo_id = f"{self.config.hf_user_id}/{detailed_repo_name}"
        
        try:
            # Push summary results (JSON file)
            if os.path.exists(self.OUTPUT_FILE_NAME_SUMMARY):
                print(f"--- 📤 Pushing summary results to {summary_repo_id} ---")
                push_dataset_to_huggingface(
                    repo_id=summary_repo_id,
                    dataset_path=self.OUTPUT_FILE_NAME_SUMMARY
                )
                print(f"--- ✅ Summary results pushed to {summary_repo_id} ---")
            else:
                print(f"--- ⚠️ Summary file {self.OUTPUT_FILE_NAME_SUMMARY} not found, skipping push ---")
            
            # Push detailed results (Excel file)
            if os.path.exists(self.OUTPUT_FILE_NAME_DETAILED):
                print(f"--- 📤 Pushing detailed results to {detailed_repo_id} ---")
                push_dataset_to_huggingface(
                    repo_id=detailed_repo_id,
                    dataset_path=self.OUTPUT_FILE_NAME_DETAILED
                )
                print(f"--- ✅ Detailed results pushed to {detailed_repo_id} ---")
            else:
                print(f"--- ⚠️ Detailed file {self.OUTPUT_FILE_NAME_DETAILED} not found, skipping push ---")
                
        except Exception as e:
            print(f"--- ❌ Error pushing results to HuggingFace Hub: {e} ---")
            raise

    def run(self, run_name: str | None = None) -> None:
        """
        Run the evaluation process.
        """
        # Login to Hugging Face
        login_huggingface()

        # Setup openai key
        setup_openai_key()

        # Setup run name
        if run_name is not None:
            self.run_name = run_name
        else:
            self.run_name = setup_run_name(
                name=self.config.run_name,
                prefix=self.config.run_name_prefix,
                suffix=self.config.run_name_suffix,
            )
        print(f"--- ✅ Run name set to: {self.run_name} ---")

        if mlflow.active_run() is not None:
            try:
                # Log configurations
                log_configurations_to_mlflow(self.config)

                # Run name logging
                mlflow.log_param("run_name", self.run_name)
            except Exception as e:
                print(f"--- ⚠️ Warning: Failed to log initial parameters: {e} ---")
                
        # Load the Ragas metrics functions based on the configuration
        self.set_ragas_metrics()
        print(f"--- ✅ Loaded Ragas metrics: {self.config.metrics} ---")

        # Load embeddings and LLM
        self.load_embeddings()
        self.load_llm()
        print(
            f"--- ✅ Loaded embeddings model: {self.config.embedding} and LLM: {self.config.llm} ---"
        )

        # Load the inferencer output dataset
        self.load_inferencer_output()
        print(
            f"--- ✅ Loaded inferencer output dataset from {self.INPUT_FILE_NAME} ---"
        )

        # Evaluate the dataset using the specified metrics
        self.evaluate()
        print(
            f"--- ✅ Evaluation completed with results: {self.get_summary_results()} ---"
        )

        # Get token count and cost
        try:
            token_count_and_cost = self.get_token_count_and_cost()
            print(f"--- 💲 Token count and cost: {token_count_and_cost} ---")
        except Exception as e:
            print(f"--- ⚠️ Could not get token count and cost: {e} ---")
            token_count_and_cost = []

        # Save results
        self.save_results()
        print(
            f"--- ✅ Results saved to {self.OUTPUT_FILE_NAME_DETAILED} and {self.OUTPUT_FILE_NAME_SUMMARY} ---"
        )

        # Push results to HuggingFace Hub
        self.push_results_to_huggingface()
        print(f"--- ✅ Results pushed to HuggingFace Hub ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the language model output")
    parser.add_argument(
        "--hf-key",
        type=str,
        required=True,
        help="Hugging Face API key for authentication",
    )
    parser.add_argument(
        "--openai-key",
        type=str,
        required=True,
        help="OpenAI API key for LLM evaluation",
    )
    args = parser.parse_args()

    # Set Hugging Face token from command line argument
    os.environ["HF_TOKEN"] = args.hf_key

    # Set OpenAI API key from command line argument
    os.environ["OPENAI_API_KEY"] = args.openai_key

    evaluator = Evaluator()
    evaluator.run()
