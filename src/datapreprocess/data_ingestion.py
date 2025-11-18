import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess both text ('Content') and label ('Label') columns.
    Includes:
    - Lowercasing
    - Whitespace cleanup
    - Label filtering + int conversion
    - Dropping Content_int column
    """
    try:
        # ---------- Drop Unwanted Column ----------
        if 'Content_int' in df.columns:
            df = df.drop(columns=['Content_int'])
            logger.debug("Dropped 'Content_int' column.")

        # ---------- Content Preprocessing ----------
        if 'Content' not in df.columns:
            raise KeyError("Column 'Content' not found in dataframe")

        # Convert to clean lowercase string
        df['Content'] = df['Content'].astype(str)
        df['Content'] = df['Content'].str.replace(r'\s+', ' ', regex=True)
        df['Content'] = df['Content'].str.strip().str.lower()

        logger.debug('Content preprocessing completed (no stemming).')

        # ---------- Label Preprocessing ----------
        if 'Label' not in df.columns:
            raise KeyError("Column 'Label' not found in dataframe")

        # Keep only valid labels
        df = df[df['Label'].astype(str).isin(['0', '1'])].copy()

        # Convert to int
        df['Label'] = df['Label'].astype(int)

        logger.debug('Label preprocessing completed.')

        return df

    except KeyError as e:
        logger.error("Missing expected column: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during preprocessing: %s", e)
        raise



def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        # data_path IS ALREADY "data/raw"
        os.makedirs(data_path, exist_ok=True)

        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)

        logger.debug('Train and test data saved to %s', data_path)
    except Exception as e:
        logger.exception('Unexpected error occurred while saving the data')
        raise



def main():
    try:
        root_dir = os.path.dirname(os.path.abspath(__file__))

        # Load parameters
        params = load_params(os.path.join(root_dir, '../../params.yaml'))
        test_size = params['data_ingestion']['test_size']

        # Load dataset
        df_path = os.path.join(root_dir, '../../dataset/HateSpeechDataset.csv')
        df = load_data(df_path)

        # Preprocess both content + labels
        final_df = preprocess(df)

        # Train-test split
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        # Save processed datasets
        save_data(train_data, test_data, os.path.join(root_dir, '../../data/raw'))

    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
