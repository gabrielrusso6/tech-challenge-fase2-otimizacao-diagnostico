from pathlib import Path
from typing import Dict, Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


class ModelEvaluator:
    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}

    def evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict[str, Any]:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        self.results[model_name] = metrics
        return metrics

    def compare_models(self) -> pd.DataFrame:
        rows = []
        for metrics in self.results.values():
            row = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
            rows.append(row)
        df = pd.DataFrame(rows)
        return df.sort_values(["f1_score", "recall", "auc_roc"], ascending=False)

    def save_comparison(self, output_path: str = "reports/model_comparison.csv") -> pd.DataFrame:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df = self.compare_models()
        df.to_csv(output_path, index=False)
        return df
