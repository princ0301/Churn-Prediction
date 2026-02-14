import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import List, Dict
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_curve, auc

logger = logging(__name__)

class VisualizationGenerator:

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style('whitegrid')

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: List[str],
        title: str,
        filename: str
    ):
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel("Predicted Label")

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusion matrix to {filepath}")

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str,
        filename: str
    ):
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved ROC curve to {filepath}")

    def plot_feature_importance(
        self, 
        model: BaseEstimator, 
        feature_names: List[str],
        filename: str,
        top_n: int = 20
    ):
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            logger.warning("Model does not support feature importance")
            return
        
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] if i < len(feature_names) else f'feature_{i}' 
                       for i in indices]
        top_importances = importances[indices]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_importances)), top_importances)
        plt.yticks(range(len(top_importances)), top_features)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved feature importance to {filepath}")

    def plot_model_comparison(
        self, 
        results: Dict[str, Dict[str, float]],
        filename: str
    ):
        models = list(results.keys())
        metrics = list(next(iter(results.values())).keys())
        
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        
        plt.figure(figsize=(12, 6))
        for i, model in enumerate(models):
            values = [results[model].get(m, 0) for m in metrics]
            plt.bar(x + i * width, values, width, label=model)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Comparison')
        plt.xticks(x + width * (len(models) - 1) / 2, metrics)
        plt.legend()
        plt.grid(axis='y')
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved model comparison to {filepath}")
