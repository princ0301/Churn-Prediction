import numpy as np
import logging
from typing import Dict
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, classification_report
from .metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)

class ModelEvaluator:

    def __init__(self, model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def compute_metrics(self) -> Dict[str, float]:
       y_pred = self.get_predictions()
       y_prob = self.get_prediction_probabilities()

       metrics = MetricsCalculator.calculate_all_mterics(
           self.y_test, y_pred, y_prob
       )
       logger.info(f"Metrics: {metrics}")
       return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        y_pred = self.get_predictions()
        return confusion_matrix(self.y_test, y_pred)
    
    def get_classification_report(self) -> str:
        y_pred = self.get_predictions()
        return classification_report(self.y_test, y_pred)
    
    def get_predictions(self) -> np.ndarray:
        return self.model.predict(self.X_test)
    
    def get_prediction_probabilities(self) -> np.ndarray:
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(self.X_test)[:, 1]
        return None
