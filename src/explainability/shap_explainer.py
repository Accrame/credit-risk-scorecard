"""SHAP explainability — global importance + per-prediction explanations."""

import numpy as np
import pandas as pd
import shap


class CreditRiskExplainer:
    """Wraps SHAP TreeExplainer for credit risk models."""

    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.expected_value = None

    def fit(self, X):
        """Fit using training data as background."""
        self.explainer = shap.TreeExplainer(self.model)
        self.expected_value = self.explainer.expected_value

        # Handle binary classification expected value
        if isinstance(self.expected_value, np.ndarray):
            if len(self.expected_value) > 1:
                self.expected_value = self.expected_value[1]
            else:
                self.expected_value = self.expected_value[0]

        return self

    def explain(self, X):
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")

        shap_values = self.explainer.shap_values(X)

        # Handle binary classification output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        self.shap_values = shap_values
        return shap_values

    def get_global_importance(self, X):
        """Mean |SHAP| per feature — global importance ranking."""
        shap_values = self.explain(X)

        # Mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)

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

        # Add percentage
        df["importance_pct"] = df["importance"] / df["importance"].sum() * 100

        return df.reset_index(drop=True)

    def explain_prediction(self, X_single, top_n=5):
        """Explain one prediction — returns risk factors and protective factors."""
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")

        # Ensure 2D
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)

        shap_values = self.explainer.shap_values(X_single)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values = shap_values.flatten()

        # Get feature names
        if self.feature_names is not None:
            names = self.feature_names
        else:
            names = [f"feature_{i}" for i in range(len(shap_values))]

        # Create contribution DataFrame
        contributions = pd.DataFrame(
            {
                "feature": names,
                "shap_value": shap_values,
                "value": X_single.flatten()[: len(names)],
            }
        )

        # Sort by absolute contribution
        contributions["abs_shap"] = np.abs(contributions["shap_value"])
        contributions = contributions.sort_values("abs_shap", ascending=False)

        # Get top positive (risk increasing) and negative (risk decreasing) factors
        risk_factors = contributions[contributions["shap_value"] > 0].head(top_n)
        protective_factors = contributions[contributions["shap_value"] < 0].head(top_n)

        # Calculate total prediction
        base_prob = 1 / (1 + np.exp(-self.expected_value))
        final_log_odds = self.expected_value + shap_values.sum()
        final_prob = 1 / (1 + np.exp(-final_log_odds))

        return {
            "base_probability": base_prob,
            "final_probability": final_prob,
            "total_shap": shap_values.sum(),
            "risk_factors": risk_factors[["feature", "shap_value", "value"]].to_dict("records"),
            "protective_factors": protective_factors[["feature", "shap_value", "value"]].to_dict(
                "records"
            ),
            "all_contributions": contributions[["feature", "shap_value", "value"]].to_dict(
                "records"
            ),
        }

    def get_adverse_action_reasons(self, X_single, top_n=4):
        """Generate denial reasons — required by ECOA. Usually top 4 risk factors."""
        explanation = self.explain_prediction(X_single, top_n=top_n)

        # Get risk factors (features that increased default probability)
        risk_factors = explanation["risk_factors"]

        # Map to human-readable reasons
        reason_templates = {
            "checking_account_status": "Insufficient checking account history",
            "credit_history": "Limited or adverse credit history",
            "credit_amount": "Requested loan amount too high",
            "duration_months": "Loan term too long",
            "age": "Length of time accounts have been established",
            "employment_years": "Insufficient employment history",
            "savings_account": "Insufficient savings or investments",
            "property": "Lack of sufficient collateral",
            "existing_credits": "Too many existing credit obligations",
            "installment_rate": "High debt-to-income ratio",
            "housing": "Housing situation",
        }

        reasons = []
        for factor in risk_factors[:top_n]:
            feature = factor["feature"]

            # Get template or create generic reason
            reason = reason_templates.get(
                feature.split("_")[0],  # Handle one-hot encoded features
                f"Information related to {feature}",
            )

            reasons.append(
                {
                    "reason_code": feature[:20].upper().replace(" ", "_"),
                    "reason_text": reason,
                    "impact": "HIGH" if abs(factor["shap_value"]) > 0.1 else "MEDIUM",
                }
            )

        return reasons

    def get_shap_summary_data(self, X):
        """Get what shap.summary_plot needs."""
        shap_values = self.explain(X)

        if self.feature_names is not None:
            names = self.feature_names
        else:
            names = [f"feature_{i}" for i in range(shap_values.shape[1])]

        return shap_values, X, names


def create_explanation_report(explainer, X_single, score):
    """Format a human-readable explanation for one applicant."""
    explanation = explainer.explain_prediction(X_single)
    reasons = explainer.get_adverse_action_reasons(X_single)

    report = []
    report.append("=" * 60)
    report.append("CREDIT DECISION EXPLANATION")
    report.append("=" * 60)
    report.append(f"\nCredit Score: {score}")
    report.append(f"Default Probability: {explanation['final_probability']:.1%}")

    report.append("\n--- Key Risk Factors ---")
    for i, factor in enumerate(explanation["risk_factors"][:4], 1):
        report.append(f"{i}. {factor['feature']}: {factor['shap_value']:.3f}")

    report.append("\n--- Positive Factors ---")
    for i, factor in enumerate(explanation["protective_factors"][:3], 1):
        report.append(f"{i}. {factor['feature']}: {factor['shap_value']:.3f}")

    report.append("\n--- Adverse Action Reasons ---")
    for i, reason in enumerate(reasons, 1):
        report.append(f"{i}. {reason['reason_text']}")

    report.append("\n" + "=" * 60)

    return "\n".join(report)


if __name__ == "__main__":
    print("SHAP Explainer module loaded successfully")
    print("Usage:")
    print("  explainer = CreditRiskExplainer(model, feature_names)")
    print("  explainer.fit(X_train)")
    print("  explanation = explainer.explain_prediction(X_test[0])")
