"""
Evaluation Testing Script
========================

This script allows you to test the evaluation functionality independently.
It loads configuration from config.toml and runs evaluation on sample data.

Usage:
    python evaluation_test.py --hf-key YOUR_HF_KEY --openai-key YOUR_OPENAI_KEY
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add the app directory to the path so we can import from it
# sys.path.append(str(Path(__file__).parent / "app"))

from evaluator import Evaluator
from config_manager import get_config_manager, MLFlowConfig


def create_sample_inference_data():
    """
    Create sample inference data for testing evaluation.
    This simulates the output from the inferencer.
    """
    sample_data = [
        {
            "system_prompt": "You are a helpful assistant that provides accurate information.",
            "user_prompt": "What is the capital of France?",
            "assistant_response": "The capital of France is Paris.",
            "ground_truth": "Paris"
        },
        {
            "system_prompt": "You are a helpful assistant that provides accurate information.",
            "user_prompt": "What is 2 + 2?",
            "assistant_response": "2 + 2 equals 4.",
            "ground_truth": "4"
        },
        {
            "system_prompt": "You are a helpful assistant that provides accurate information.",
            "user_prompt": "What is the largest planet in our solar system?",
            "assistant_response": "Jupiter is the largest planet in our solar system.",
            "ground_truth": "Jupiter"
        },
        {
            "system_prompt": "You are a helpful assistant that provides accurate information.",
            "user_prompt": "Who wrote Romeo and Juliet?",
            "assistant_response": "William Shakespeare wrote Romeo and Juliet.",
            "ground_truth": "William Shakespeare"
        },
        {
            "system_prompt": "You are a helpful assistant that provides accurate information.",
            "user_prompt": "What is the chemical symbol for water?",
            "assistant_response": "The chemical symbol for water is H2O.",
            "ground_truth": "H2O"
        },
        {
            "system_prompt": "You are a helpful assistant that provides accurate information.",
            "user_prompt": "What year did World War II end?",
            "assistant_response": "World War II ended in 1945.",
            "ground_truth": "1945"
        },
        {
            "system_prompt": "You are a helpful assistant that provides accurate information.",
            "user_prompt": "What is the speed of light?",
            "assistant_response": "The speed of light in vacuum is approximately 299,792,458 meters per second.",
            "ground_truth": "299,792,458 m/s"
        },
        {
            "system_prompt": "You are a helpful assistant that provides accurate information.",
            "user_prompt": "What is the smallest unit of matter?",
            "assistant_response": "The atom is considered the smallest unit of matter that retains the properties of an element.",
            "ground_truth": "atom"
        },
        {
            "system_prompt": "You are a helpful assistant that provides accurate information.",
            "user_prompt": "What is the capital of Japan?",
            "assistant_response": "Tokyo is the capital of Japan.",
            "ground_truth": "Tokyo"
        },
        {
            "system_prompt": "You are a helpful assistant that provides accurate information.",
            "user_prompt": "What is photosynthesis?",
            "assistant_response": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen.",
            "ground_truth": "The process by which plants convert light energy into chemical energy (glucose) using CO2 and water"
        }
    ]
    
    # Save to JSONL file
    with open("inferencer_output.jsonl", "w") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"--- ✅ Created sample inference data with {len(sample_data)} examples ---")
    return len(sample_data)

def print_config_info():
    """
    Print information about the current configuration.
    """
    config_manager = get_config_manager()
    evaluator_config = config_manager.get_section("evaluator")
    
    print("--- 📋 Current Evaluator Configuration ---")
    print(f"Metrics: {evaluator_config.get('metrics', [])}")
    print(f"LLM: {evaluator_config.get('llm', 'Not set')}")
    print(f"Embedding: {evaluator_config.get('embedding', 'Not set')}")
    print(f"Run name: {evaluator_config.get('run_name', 'Not set')}")
    print(f"Run name prefix: {evaluator_config.get('run_name_prefix', 'Not set')}")
    print(f"Run name suffix: {evaluator_config.get('run_name_suffix', 'Not set')}")


def cleanup_test_files():
    """
    Clean up test files created during testing.
    """
    test_files = [
        "inferencer_output.jsonl",
        "evaluator_output_detailed.xlsx",
        "evaluator_output_summary.json"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"--- 🧹 Cleaned up {file} ---")




def main():
    """
    Main function to run the evaluation test.
    """
    parser = argparse.ArgumentParser(description="Test the evaluation functionality")
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
    parser.add_argument(
        "--cleanup",    
        action="store_true",
        help="Clean up test files after running",
    )
    parser.add_argument(
        "--skip-sample-data",
        action="store_true",
        help="Skip creating sample data (use existing inferencer_output.jsonl)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Custom run name for this evaluation test",
    )
    
    args = parser.parse_args()
    
    print("=== 🧪 EVALUATION TESTING SCRIPT ===")
    print("This script will test the evaluation functionality independently.")
    print()
    
    # Set environment variables
    os.environ["HF_TOKEN"] = args.hf_key
    os.environ["OPENAI_API_KEY"] = args.openai_key
    
    try:
        # Print configuration info
        print_config_info()
        print()
        
        # Create sample data if not skipping
        if not args.skip_sample_data:
            sample_count = create_sample_inference_data()
            print(f"--- 📝 Using {sample_count} sample inference examples ---")
        else:
            if not os.path.exists("inferencer_output.jsonl"):
                print("--- ❌ inferencer_output.jsonl not found ---")
                print("Either remove --skip-sample-data or provide an existing inferencer_output.jsonl file")
                return 1
            print("--- 📝 Using existing inferencer_output.jsonl ---")
        
        print()
        
        # Initialize and run evaluator
        print("--- 🚀 Starting evaluation process ---")
        evaluator = Evaluator()
        
        # Run evaluation with custom run name if provided
        if args.run_name:
            run_name = args.run_name
            print(f"--- 🏷️ Using run name provided in CLI Args: {args.run_name} ---")
        else:
            config = MLFlowConfig.from_config(get_config_manager())
            print(f"--- 🏷️ Using run name from config: {config.run_name} ---")
            run_name = config.run_name
        run_name = f"{run_name}_evaluation" 
        evaluator.run(run_name=run_name)
        
        print()
        print("--- 🎉 Evaluation completed successfully! ---")
        print("Check the following files for results:")
        print("  - evaluator_output_detailed.xlsx (detailed results)")
        print("  - evaluator_output_summary.json (summary results)")
        
        # Print token usage if available
        if hasattr(evaluator, 'get_token_count_and_cost'):
            try:
                token_info = evaluator.get_token_count_and_cost()
                print(f"--- 💰 Token Usage: {token_info} ---")
            except Exception as e:
                print(f"--- ⚠️ Could not get token usage: {e} ---")
        
        # Cleanup if requested
        if args.cleanup:
            print()
            print("--- 🧹 Cleaning up test files ---")
            cleanup_test_files()
        
        return 0
        
    except Exception as e:
        print(f"--- ❌ Evaluation test failed: {e} ---")
        print("Please check your configuration and try again.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
