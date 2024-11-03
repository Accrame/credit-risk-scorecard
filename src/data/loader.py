"""Data loading for the German Credit Dataset."""

from pathlib import Path

import pandas as pd
import requests

# German Credit Dataset column names (the UCI dataset has no headers)
COLUMN_NAMES = [
    "checking_account_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_years",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_years",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "credit_risk",  # Target: 1 = Good, 2 = Bad
]

# Mapping for categorical variables (German Credit uses codes)
CATEGORICAL_MAPPINGS = {
    "checking_account_status": {
        "A11": "< 0 DM",
        "A12": "0-200 DM",
        "A13": ">= 200 DM",
        "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits taken / all paid back duly",
        "A31": "all credits at this bank paid back duly",
        "A32": "existing credits paid back duly till now",
        "A33": "delay in paying off in the past",
        "A34": "critical account / other credits existing",
    },
    "purpose": {
        "A40": "car (new)",
        "A41": "car (used)",
        "A42": "furniture/equipment",
        "A43": "radio/television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "others",
    },
    "savings_account": {
        "A61": "< 100 DM",
        "A62": "100-500 DM",
        "A63": "500-1000 DM",
        "A64": ">= 1000 DM",
        "A65": "unknown / no savings account",
    },
    "employment_years": {
        "A71": "unemployed",
        "A72": "< 1 year",
        "A73": "1-4 years",
        "A74": "4-7 years",
        "A75": ">= 7 years",
    },
    "personal_status_sex": {
        "A91": "male: divorced/separated",
        "A92": "female: divorced/separated/married",
        "A93": "male: single",
        "A94": "male: married/widowed",
        "A95": "female: single",
    },
    "other_debtors": {
        "A101": "none",
        "A102": "co-applicant",
        "A103": "guarantor",
    },
    "property": {
        "A121": "real estate",
        "A122": "building society savings / life insurance",
        "A123": "car or other",
        "A124": "unknown / no property",
    },
    "other_installment_plans": {
        "A141": "bank",
        "A142": "stores",
        "A143": "none",
    },
    "housing": {
        "A151": "rent",
        "A152": "own",
        "A153": "for free",
    },
    "job": {
        "A171": "unemployed / unskilled - non-resident",
        "A172": "unskilled - resident",
        "A173": "skilled employee / official",
        "A174": "management / self-employed / highly qualified",
    },
    "telephone": {
        "A191": "none",
        "A192": "yes, registered",
    },
    "foreign_worker": {
        "A201": "yes",
        "A202": "no",
    },
}


class CreditDataLoader:
    """Loads the German Credit Dataset from UCI (or cache)."""

    UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.raw_path = self.data_dir / "german_credit.csv"

    def fetch_data(self, force_download=False):
        """Download from UCI or load cached CSV."""
        if self.raw_path.exists() and not force_download:
            return pd.read_csv(self.raw_path)

        # Download from UCI
        response = requests.get(self.UCI_URL, timeout=30)
        response.raise_for_status()

        # Parse the space-separated file
        lines = response.text.strip().split("\n")
        data = [line.split() for line in lines]

        df = pd.DataFrame(data, columns=COLUMN_NAMES)

        # Convert numeric columns
        numeric_cols = [
            "duration_months",
            "credit_amount",
            "installment_rate",
            "residence_years",
            "age",
            "existing_credits",
            "num_dependents",
            "credit_risk",
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])

        # Convert target: 1 = Good (0), 2 = Bad (1) -> we predict default (bad)
        df["target"] = (df["credit_risk"] == 2).astype(int)
        df = df.drop("credit_risk", axis=1)

        # Save to cache
        self.data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.raw_path, index=False)

        return df

    def load_with_labels(self, force_download=False):
        """Load data and decode the A11/A12/etc codes to readable labels."""
        df = self.fetch_data(force_download)

        # Decode categorical columns
        for col, mapping in CATEGORICAL_MAPPINGS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(df[col])

        return df

    def get_train_test_split(self, test_size=0.2, random_state=42):
        from sklearn.model_selection import train_test_split

        df = self.fetch_data()

        X = df.drop("target", axis=1)
        y = df["target"]

        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


if __name__ == "__main__":
    # Quick test
    loader = CreditDataLoader()
    df = loader.fetch_data()
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    print(f"Default rate: {df['target'].mean():.2%}")
