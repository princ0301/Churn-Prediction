import pandas as pd
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class DataSplitter:
    
    @staticmethod
    def split_data(
        df: pd.DataFrame, 
        target: str, 
        test_size: float, 
        random_state: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        
        X = df.drop(columns=[target])
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        
        logger.info(f"Split data: train={len(X_train)}, test={len(X_test)}")
        return X_train, X_test, y_train, y_test
