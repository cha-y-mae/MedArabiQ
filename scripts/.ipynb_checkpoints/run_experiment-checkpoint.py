import os
import json
import pandas as pd
import logging
from tqdm import tqdm  # Import tqdm for progress bar
from scripts.utils import load_config, save_predictions
from models import load_model_handler
from evaluations.evaluator import evaluate        # Import the evaluator

logging.basicConfig(level=logging.INFO)           # Configure logging

def run_experiment(config_path):
    logging.info(f"Loading config from {config_path}")
    config = load_config(config_path)  # Load config file 
    logging.info(f"Initializing model: {config['model']['name']}")
    
    # Load the appropriate model handler
    model_handler = load_model_handler(config)
    logging.info(f"Loading dataset from {config['dataset']['path']}")
    dataset_path = config['dataset']['path']
    instruction_path = config['dataset'].get('instruction_path')
    task_type = config['task']['type']

    # Load dataset with utf-8 encoding
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    with open(instruction_path, 'r') as f:
        instruction = f.read().strip()

    predictions = []

     # Generate predictions with a progress bar
    for idx, item in enumerate(tqdm(dataset, desc="Processing examples"), start=1):
        input_text = item.get('Question')
        if not input_text:
            logging.warning(f"No input text found for item {idx}. Skipping.")
            continue

        # Pass instruction as system_prompt
        prediction = model_handler.prompt(input_text, instruction, task_type)

        predictions.append({
            "id": idx,
            "input": input_text,
            "prediction": prediction,
            "ground_truth": item.get('Answer')
        })

    # Save predictions to file
    output_path = config['output']['predictions_path']
    logging.info(f"Saving predictions to {output_path}")
    save_predictions(predictions, output_path)


    # Perform evaluation
    metrics_path = config['output']['metrics_path']
    task_type = config['task']['type']
    logging.info("Starting evaluation...")
    metrics = evaluate(output_path, metrics_path, task_type)
    logging.info(f"Evaluation completed. Metrics saved to {metrics_path}")
    logging.info(f"Metrics: {json.dumps(metrics, indent=4)}")

# Run the script
if __name__ == "__main__":
    import sys
    config_file = sys.argv[1]  # Give config file as a command-line argument
    try:
        run_experiment(config_file)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
