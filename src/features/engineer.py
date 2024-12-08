"""Feature engineering — derived features based on credit risk domain knowledge."""

import pandas as pd


class CreditFeatureEngineer:
    """Creates derived features like debt ratios, stability scores, risk flags."""

    def __init__(self):
        self.fitted = False
        self.feature_stats = {}

    def fit(self, X, y=None):
        # Store statistics for later use
        if "credit_amount" in X.columns:
            self.feature_stats["credit_amount_median"] = X["credit_amount"].median()
        if "age" in X.columns:
            self.feature_stats["age_median"] = X["age"].median()

        self.fitted = True
        return self

    def transform(self, X):
        df = X.copy()

        # Debt burden ratio: credit amount relative to duration
        if "credit_amount" in df.columns and "duration_months" in df.columns:
            df["monthly_payment"] = df["credit_amount"] / df["duration_months"]
            df["debt_burden"] = df["credit_amount"] / (df["duration_months"] * 100)

        # Credit amount to income proxy (using installment rate as proxy)
        if "credit_amount" in df.columns and "installment_rate" in df.columns:
            # Installment rate is 1-4, representing % of disposable income
            df["credit_to_income"] = df["credit_amount"] / (5 - df["installment_rate"])

        # Age-based features
        if "age" in df.columns:
            df["age_group"] = pd.cut(
                df["age"],
                bins=[0, 25, 35, 45, 55, 100],
                labels=["young", "early_career", "mid_career", "senior", "retired"],
            )
            df["is_young"] = (df["age"] < 25).astype(int)
            df["is_senior"] = (df["age"] > 55).astype(int)

        # Stability score (combines employment and residence)
        if "employment_years" in df.columns and "residence_years" in df.columns:
            employment_map = {"A71": 0, "A72": 1, "A73": 2, "A74": 3, "A75": 4}
            if df["employment_years"].dtype == "object":
                emp_score = df["employment_years"].map(employment_map).fillna(2)
            else:
                emp_score = df["employment_years"]

            df["stability_score"] = emp_score + df["residence_years"]

        # Has guarantor or co-applicant
        if "other_debtors" in df.columns:
            df["has_support"] = (df["other_debtors"] != "A101").astype(int)
            if df["other_debtors"].dtype == "object":
                df["has_support"] = (~df["other_debtors"].isin(["A101", "none"])).astype(int)

        # Property risk (no property = higher risk)
        if "property" in df.columns:
            high_risk_property = ["A124", "unknown / no property"]
            df["no_property"] = df["property"].isin(high_risk_property).astype(int)

        # Checking account risk
        if "checking_account_status" in df.columns:
            no_account = ["A14", "no checking account"]
            df["no_checking"] = df["checking_account_status"].isin(no_account).astype(int)

        # Credit history risk
        if "credit_history" in df.columns:
            risky_history = [
                "A30",
                "A31",
                "no credits taken / all paid back duly",
                "all credits at this bank paid back duly",
            ]
            # Paradoxically, "no credits" can be risky (no track record)
            df["limited_history"] = df["credit_history"].isin(risky_history).astype(int)

        # Multiple credits indicator
        if "existing_credits" in df.columns:
            df["multiple_credits"] = (df["existing_credits"] > 1).astype(int)

        # High amount flag (above median)
        if "credit_amount" in df.columns and "credit_amount_median" in self.feature_stats:
            median = self.feature_stats["credit_amount_median"]
            df["high_amount"] = (df["credit_amount"] > median).astype(int)

        # Long duration flag
        if "duration_months" in df.columns:
            df["long_duration"] = (df["duration_months"] > 24).astype(int)

        return df

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_derived_feature_names(self):
        return [
            "monthly_payment",
            "debt_burden",
            "credit_to_income",
            "age_group",
            "is_young",
            "is_senior",
            "stability_score",
            "has_support",
            "no_property",
            "no_checking",
            "limited_history",
            "multiple_credits",
            "high_amount",
            "long_duration",
        ]


def create_interaction_features(df):
    """Multiply risk flags together to capture compound risk."""
    result = df.copy()

    # High risk interactions
    if "is_young" in df.columns and "high_amount" in df.columns:
        result["young_high_amount"] = df["is_young"] * df["high_amount"]

    if "no_checking" in df.columns and "high_amount" in df.columns:
        result["no_checking_high_amount"] = df["no_checking"] * df["high_amount"]

    if "limited_history" in df.columns and "long_duration" in df.columns:
        result["new_borrower_long_loan"] = df["limited_history"] * df["long_duration"]

    return result


if __name__ == "__main__":
    # Quick test
    from src.data.loader import CreditDataLoader

    loader = CreditDataLoader()
    df = loader.fetch_data()
    X = df.drop("target", axis=1)

    engineer = CreditFeatureEngineer()
    X_engineered = engineer.fit_transform(X)

    print(f"Original features: {len(X.columns)}")
    print(f"After engineering: {len(X_engineered.columns)}")
    print(f"New features: {set(X_engineered.columns) - set(X.columns)}")
