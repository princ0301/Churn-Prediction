import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

class HyperparameterTuner:

    def __init__(
        self,
        model: BaseEstimator,
        param_grid: Dict[str, List],
        cv_folds: int = 5,
        scoring: str = 'roc_auc'
    ):
        self.model = model
        self.param_grid = param_grid
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.grid_search = None

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        
        self.grid_search = GridSearchCV(
            self.model,
            self.param_grid,
            cv=self.cv_folds,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=1
        )

        self.grid_search.fit(X_train, y_train)

        logger.info(f"Best params: {self.grid_search.best_params_}")
        logger.info(f"Best CV score: {self.grid_search.best_score_:.4f}")

        return self.grid_search.best_estimator_, self.grid_search.best_params_
    
    def get_cv_results(self) -> pd.DataFrame:
        if self.grid_search is None:
            return pd.DataFrame()
        return pd.DataFrame(self.grid_search.cv_results_)