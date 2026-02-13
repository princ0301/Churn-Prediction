import pandas as pd
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class DataValidator:

    @staticmethod
    def check_missing_values(df: pd.DataFrame) -> Dict[str, float]:
        missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
        missing_pct = {k: v for k, v in missing_pct.items() if v > 0}

        if missing_pct:
            logger.warning(f"Found missing values in {len(missing_pct)} columns")
        return missing_pct
    
    @staticmethod
    def check_required_columns(df: pd.DataFrame, required: List[str]) -> bool:
        missing_cols = [col for col in required if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        return True
    
    @staticmethod
    def check_data_types(df: pd.DataFrame, expexted_type: Dict[str, str]) -> bool:
        for col, expexted_types in expexted_types.items():
            if col not in df.columns:
                continue
            actual_type = str(df[col].dtype)
            if expexted_type not in actual_type:
                raise TypeError(f"Column '{col}' has type {actual_type}, expected {expexted_type}")
        return True
    
    @staticmethod
    def generate_data_report(df: pd.DataFrame) -> Dict[str, Any]:
        report = {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'missing_values': DataValidator.check_missing_values(df),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True)
        }
        logger.info(f"Generated data report: {report['n_samples']} samples, {report['n_features']} features")
        return report