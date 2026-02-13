import pandas as pd
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

class DataLoader:
    
    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from {filepath}")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")
    
    @staticmethod
    def get_feature_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
        features = [col for col in df.columns if col != target]
        numeric_features = df[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df[features].select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Identified {len(numeric_features)} numeric and {len(categorical_features)} categorical features")
        return numeric_features, categorical_features

if __name__ == "__main__":
    df = DataLoader.load_csv("D:/Project_new/Churn_Prediction/data/dataset.csv")
    print(df)

    numeric, categorical = DataLoader.get_feature_types(df, target="Churn")

    print("Numeric Features:", numeric)
    print("Categorical Features:", categorical)