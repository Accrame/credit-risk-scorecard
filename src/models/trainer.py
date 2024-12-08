"""Model training and evaluation for credit risk scoring."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


class CreditRiskTrainer:
    """Trains and evaluates credit risk models with XGBoost."""

    # didn't tune these much, mostly defaults + some regularization
    DEFAULT_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 4,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(self, params=None):
        self.model = None
        self.feature_names = None
        self.metrics = {}
        self.params = params or self.DEFAULT_PARAMS.copy()

    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        """Train the XGBoost model."""
        self.feature_names = feature_names
        self.model = xgb.XGBClassifier(**self.params)

        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train)

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test, threshold=0.5):
        """Evaluate model with banking-specific metrics."""
        y_proba = self.predict_proba(X_test)
        y_pred = (y_proba >= threshold).astype(int)

        # Basic metrics
        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y_test, y_proba),
        }

        # Gini coefficient (2 * AUC - 1)
        self.metrics["gini"] = 2 * self.metrics["auc_roc"] - 1

        # KS statistic
        self.metrics["ks_statistic"] = self._calculate_ks(y_test, y_proba)

        # Precision at various thresholds
        for pct in [5, 10, 20]:
            self.metrics[f"precision_at_{pct}pct"] = self._precision_at_k(
                y_test, y_proba, pct / 100
            )

        return self.metrics

    def _calculate_ks(self, y_true, y_proba):
        """KS statistic — max separation between cumulative good/bad distributions."""
        # Sort by probability descending
        sorted_idx = np.argsort(-y_proba)
        y_sorted = y_true[sorted_idx]

        # Calculate cumulative distributions
        n_pos = y_true.sum()
        n_neg = len(y_true) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.0

        cum_pos = np.cumsum(y_sorted) / n_pos
        cum_neg = np.cumsum(1 - y_sorted) / n_neg

        return np.max(np.abs(cum_pos - cum_neg))

    def _precision_at_k(self, y_true, y_proba, k):
        """Precision in top k% of predictions (useful for portfolio cutoff decisions)."""
        n_top = int(len(y_true) * k)
        if n_top == 0:
            return 0.0

        top_idx = np.argsort(-y_proba)[:n_top]
        return y_true[top_idx].mean()

    def cross_validate(self, X, y, n_splits=5):
        """Stratified k-fold CV. Returns mean/std AUC."""
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        model = xgb.XGBClassifier(**self.params)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

        return {
            "cv_auc_mean": scores.mean(),
            "cv_auc_std": scores.std(),
            "cv_scores": scores.tolist(),
        }

    def get_feature_importance(self, top_n=20):
        """Get feature importance from trained model."""
        importance = self.model.feature_importances_

        if self.feature_names is not None:
            names = self.feature_names
        else:
            names = [f"feature_{i}" for i in range(len(importance))]

        df = pd.DataFrame(
            {
                "feature": names,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)

        return df.head(top_n)

    def save(self, path):
        """Save model + metadata as joblib + json."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(self.model, path)

        metadata = {
            "params": self.params,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
        }

        metadata_path = path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load(self, path):
        """Load a saved model."""
        path = Path(path)

        self.model = joblib.load(path)

        metadata_path = path.with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

            self.params = metadata.get("params", {})
            self.feature_names = metadata.get("feature_names")
            self.metrics = metadata.get("metrics", {})

        return self


def print_evaluation_report(metrics):
    """Print evaluation report."""
    print("\n" + "=" * 50)
    print("MODEL EVALUATION REPORT")
    print("=" * 50)

    print("\nDiscrimination Metrics:")
    print(f"  AUC-ROC:      {metrics.get('auc_roc', 0):.4f}")
    print(f"  Gini:         {metrics.get('gini', 0):.4f}")
    print(f"  KS Statistic: {metrics.get('ks_statistic', 0):.4f}")

    print("\nClassification Metrics (at 0.5 threshold):")
    print(f"  Accuracy:     {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision:    {metrics.get('precision', 0):.4f}")
    print(f"  Recall:       {metrics.get('recall', 0):.4f}")
    print(f"  F1-Score:     {metrics.get('f1', 0):.4f}")

    print("\nRanking Metrics:")
    print(f"  Precision @5%:  {metrics.get('precision_at_5pct', 0):.4f}")
    print(f"  Precision @10%: {metrics.get('precision_at_10pct', 0):.4f}")
    print(f"  Precision @20%: {metrics.get('precision_at_20pct', 0):.4f}")

    print("=" * 50)


if __name__ == "__main__":
    # Quick test
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate
    trainer = CreditRiskTrainer(model_type="xgboost")
    trainer.train(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test)

    print_evaluation_report(metrics)
