"""Fairness auditing with Fairlearn — checks demographic parity, equalized odds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    selection_rate,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class FairnessAuditor:
    """Audits model predictions for bias across protected groups."""

    def __init__(self, sensitive_features):
        self.sensitive_features = sensitive_features
        self.results: dict[str, pd.DataFrame] = {}

    def audit(self, y_true, y_pred, sensitive_data):
        """Run fairness audit. Returns per-group metrics + disparity measures."""
        results = {}

        for feature in self.sensitive_features:
            if feature not in sensitive_data.columns:
                continue

            sensitive_values = sensitive_data[feature].values

            # Calculate metrics by group
            metric_frame = MetricFrame(
                metrics={
                    "selection_rate": selection_rate,
                    "accuracy": accuracy_score,
                    "precision": lambda y_t, y_p: precision_score(y_t, y_p, zero_division=0),
                    "recall": lambda y_t, y_p: recall_score(y_t, y_p, zero_division=0),
                    "f1": lambda y_t, y_p: f1_score(y_t, y_p, zero_division=0),
                },
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_values,
            )

            # Store per-group metrics
            self.results[feature] = metric_frame.by_group

            # Calculate disparity metrics
            results[feature] = {
                "by_group": metric_frame.by_group.to_dict(),
                "demographic_parity_difference": demographic_parity_difference(
                    y_true, y_pred, sensitive_features=sensitive_values
                ),
                "demographic_parity_ratio": demographic_parity_ratio(
                    y_true, y_pred, sensitive_features=sensitive_values
                ),
                "equalized_odds_difference": equalized_odds_difference(
                    y_true, y_pred, sensitive_features=sensitive_values
                ),
                "equalized_odds_ratio": equalized_odds_ratio(
                    y_true, y_pred, sensitive_features=sensitive_values
                ),
            }

            # Assess compliance
            results[feature]["compliance"] = self._assess_compliance(results[feature])

        return results

    def _assess_compliance(self, metrics):
        """Check the 80% rule (four-fifths rule) and common thresholds."""
        return {
            "four_fifths_rule": metrics["demographic_parity_ratio"] >= 0.8,
            "demographic_parity_ok": abs(metrics["demographic_parity_difference"]) < 0.1,
            "equalized_odds_ok": abs(metrics["equalized_odds_difference"]) < 0.1,
        }

    def audit_probabilities(self, y_true, y_proba, sensitive_data, threshold=0.5):
        """Like audit() but takes probabilities instead of binary predictions."""
        y_pred = (y_proba >= threshold).astype(int)
        results = self.audit(y_true, y_pred, sensitive_data)

        # Add probability distribution analysis
        for feature in self.sensitive_features:
            if feature not in sensitive_data.columns:
                continue

            proba_by_group = {}
            for group in sensitive_data[feature].unique():
                mask = sensitive_data[feature] == group
                proba_by_group[str(group)] = {
                    "mean": float(y_proba[mask].mean()),
                    "std": float(y_proba[mask].std()),
                    "median": float(np.median(y_proba[mask])),
                }

            results[feature]["probability_distribution"] = proba_by_group

        return results

    def find_fair_threshold(
        self, y_true, y_proba, sensitive_data, feature, metric="demographic_parity"
    ):
        """Find per-group thresholds that equalize selection rates.
        TODO: this is a rough approach, could use proper optimization instead."""
        if feature not in sensitive_data.columns:
            raise ValueError(f"Feature {feature} not in sensitive_data")

        groups = sensitive_data[feature].unique()

        # First, find overall target selection rate
        target_rate = (y_proba >= 0.5).mean()

        # Find threshold for each group that achieves target selection rate
        thresholds = {}
        for group in groups:
            mask = sensitive_data[feature] == group
            group_proba = y_proba[mask]

            # Binary search for threshold
            threshold = np.percentile(group_proba, (1 - target_rate) * 100)
            thresholds[str(group)] = float(threshold)

        return thresholds

    def get_summary_report(self) -> str:
        """Generate a summary fairness report."""
        if not self.results:
            return "No audit results available. Run audit() first."

        lines = []
        lines.append("=" * 60)
        lines.append("FAIRNESS AUDIT REPORT")
        lines.append("=" * 60)

        for feature, df in self.results.items():
            lines.append(f"\n--- {feature} ---")
            lines.append(df.to_string())
            lines.append("")

        return "\n".join(lines)


def create_age_groups(ages: np.ndarray) -> np.ndarray:
    """
    Create age groups for fairness analysis.

    Common protected categories:
    - Young (under 25): Protected under ECOA
    - Senior (over 62): Protected under ECOA
    """
    groups = np.where(ages < 25, "under_25", np.where(ages >= 62, "over_62", "25_to_61"))
    return groups


def extract_gender_from_personal_status(status: pd.Series) -> pd.Series:
    """
    Extract gender from German Credit personal_status_sex field.

    Note: Gender-based discrimination is prohibited in credit decisions
    in most jurisdictions. This is for audit purposes only.
    """
    gender_map = {
        "A91": "male",
        "A92": "female",
        "A93": "male",
        "A94": "male",
        "A95": "female",
        "male: divorced/separated": "male",
        "female: divorced/separated/married": "female",
        "male: single": "male",
        "male: married/widowed": "male",
        "female: single": "female",
    }
    return status.map(gender_map).fillna("unknown")


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    n = 1000

    # Synthetic data
    y_true = np.random.binomial(1, 0.3, n)
    y_pred = np.random.binomial(1, 0.3, n)
    sensitive_data = pd.DataFrame(
        {
            "age_group": np.random.choice(["young", "middle", "senior"], n),
            "gender": np.random.choice(["male", "female"], n),
        }
    )

    # Audit
    auditor = FairnessAuditor(["age_group", "gender"])
    results = auditor.audit(y_true, y_pred, sensitive_data)

    print(auditor.get_summary_report())

    print("\nCompliance Check:")
    for feature, data in results.items():
        print(f"\n{feature}:")
        for check, passed in data["compliance"].items():
            status = "PASS" if passed else "FAIL"
            print(f"  {check}: {status}")
