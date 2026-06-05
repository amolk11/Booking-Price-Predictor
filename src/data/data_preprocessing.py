from lightgbm import train
import pandas as pd
import numpy as np

from utils import logger

logger = logger.get_logger()

def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded data as a DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error occurred while loading data from {file_path}: {e}")
        raise


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data by handling missing values and encoding categorical variables.

    Args:
        data (pd.DataFrame): The raw data to preprocess.
    
    Returns:
        pd.DataFrame: The preprocessed data.
    """
    try:
        # Remove unnecessary columns
        data.drop(['Unnamed: 0', 'key', 'passenger_count'], axis=1, inplace=True)

        # Check missing values
        data.isnull().sum()
        
        # Drop rows with missing values
        data.dropna(inplace=True)
        
        return data
    except Exception as e:
        logger.error(f"Error occurred during data preprocessing: {e}")
        raise
    
def split_data(data: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    """Splits the data into training and testing sets.

    Args:
        data (pd.DataFrame): The preprocessed data to split.
        target_column (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the randomness of the split.
    Returns:
        train_data (pd.DataFrame): The training data.
        test_data (pd.DataFrame): The testing data.
    """

    try:
        from sklearn.model_selection import train_test_split
        
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        train_df , test_df = train_test_split(data, test_size=test_size, random_state=random_state)        
        logger.info(f"Data split into training and testing sets with test size {test_size}")
        
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error occurred during data splitting: {e}")
        raise
    

def save_preprocessed_data(data: pd.DataFrame, file_path: str):
    """Saves the preprocessed data to a CSV file.

    Args:
        data (pd.DataFrame): The preprocessed data to save.
        file_path (str): The path to save the CSV file.
    """
    try:
        data.to_csv(file_path, index=False)
        logger.info(f"Preprocessed data saved successfully to {file_path}")
        
    except Exception as e:
        logger.error(f"Error occurred while saving preprocessed data to {file_path}: {e}")
        raise
    

def main():
    
    # Load data
    data = load_data('data/raw/uber.csv')
    
    # Preprocess data
    preprocessed_data = preprocess_data(data)
    
    # Split data into training and testing sets
    train_data, test_data = split_data(preprocessed_data, target_column='target')
    
    # Save preprocessed data
    save_preprocessed_data(train_data, 'data/processed/train_data.csv')
    save_preprocessed_data(test_data, 'data/processed/test_data.csv')