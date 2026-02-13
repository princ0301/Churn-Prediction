import numpy as np
import logging
import time
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

class ModelTrainer:

    def __init__(self, model: BaseEstimator):
        self.model = model
        self.training_time = 0

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> BaseEstimator:
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        logger.info(f"Model trained in {self.training_time:.2f} seconds")
        return self.model
    
    def get_training_time(self) -> float:
        return self.training_time