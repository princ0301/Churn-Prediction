import pandas as pd
import numpy as np
import joblib
import logging
from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

class PreprocessingPipeline:

    def __init__(
        self,
        numeric_features: List[str],
        categorical_features: List[str],
        numeric_strategy: str = 'mean',
        categorical_strategy: str = 'onehot',
        scaling: str = 'standard'
    ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.scaling = scaling
        self.transformer = None
        self._build_transformer()

    def _build_transformer(self):
        numric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.numeric_strategy)),
            ('scaler', StandardScaler() if self.scaling == 'standard' else MinMaxScaler() if self.scaling == 'minmax' else 'passthrough')
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.transformer = ColumnTransformer(
            transformers=[
                ('num', numric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )

    def fit(self, X: pd.DataFrame):
        self.transformer.fit(X)
        logger.info("Preprocessing pipeline fitted")
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.transformer.transform(X)
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        result = self.transformer.fit_transform(X)
        logger.info(f"Transformed data shape: {result.shape}")
        return result
    
    def get_feature_names(self) -> List[str]:
        try:
            return self.transformer.get_feature_names_out().tolist()
        except:
            return []
        
    def save(self, filepath: str):
        joblib.dump(self.transformer, filepath)
        logger.info(f"Saved preprocessing pipeline to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        transformer = joblib.load(filepath)
        logger.info(f"Saved preprocessing pipeline to {filepath}")

