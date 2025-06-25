import yaml
import pandas as pd

def load_config(config_path):
    """
    Load a YAML configuration file.
    Args:
        config_path (str): Path to the YAML configuration file.
    Returns:
        dict: Parsed configuration as a dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML config file: {e}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred while loading the config: {e}")


def save_predictions(predictions, output_path):
    """
    Save predictions to a CSV file.
    Args:
        predictions (list of dict): List of dictionaries containing predictions.
        output_path (str): Path where the predictions CSV will be saved.
    Returns:
        None
    """
    try:
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(output_path, index=False)
    except Exception as e:
        raise IOError(f"Failed to save predictions to {output_path}: {e}")



