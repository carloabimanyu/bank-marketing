import os
import yaml
import joblib

file_dir = os.path.dirname(__file__)
config_dir = os.path.join(file_dir, '../config/config.yaml')

def load_config() -> dict:
    # Try to load YAML file
    try:
        with open(config_dir, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise RuntimeError("Parameters file not found in path.")
    
    # Return config in dict format
    return config

def pickle_load(file_path: str):
    # Load and return pickle file
    return joblib.load(file_path)

def pickle_dump(data, file_path: str) -> None:
    # Dump data into file
    joblib.dump(data, file_path)