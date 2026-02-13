import logging
from typing import Dict, Any, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

class ModelFactory:

    @staticmethod
    def create_model(model_name: str, **kwargs) -> BaseEstimator:
        models = {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'xgboost': XGBClassifier
        }

        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
        
        model = models[model_name](**kwargs)
        logger.info(f"Created {model_name} model")
        return model
    
    @staticmethod
    def get_param_grid(model_name: str, config: Dict[str, Any]) -> Dict[str, List]:
        if 'hyperparameters' not in config:
            return {}
        return config['hyperparameters']
