"""Preprocessing pipeline — handles encoding, scaling, imputation."""

from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class CreditDataPreprocessor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible preprocessor. Auto-detects column types if not specified."""

    def __init__(self, categorical_cols=None, numerical_cols=None, scale_numerical=True):
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.scale_numerical = scale_numerical
        self.preprocessor: ColumnTransformer | None = None
        self.feature_names_: list[str] | None = None

    def _infer_column_types(self, X):
        """Guess which columns are categorical vs numerical."""
        categorical = []
        numerical = []

        for col in X.columns:
            if X[col].dtype == "object" or X[col].nunique() < 10:
                categorical.append(col)
            else:
                numerical.append(col)

        return categorical, numerical

    def fit(self, X, y=None):
        # Infer column types if not provided
        if self.categorical_cols is None or self.numerical_cols is None:
            self.categorical_cols, self.numerical_cols = self._infer_column_types(X)

        # Build preprocessing pipelines
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        if self.scale_numerical:
            numerical_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
        else:
            numerical_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                ]
            )

        # Combine into ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", categorical_pipeline, self.categorical_cols),
                ("num", numerical_pipeline, self.numerical_cols),
            ],
            remainder="drop",
        )

        self.preprocessor.fit(X)

        # Store feature names for later use
        self._compute_feature_names()

        return self

    def _compute_feature_names(self):
        feature_names = []

        # Categorical features (one-hot encoded)
        if self.categorical_cols:
            cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["encoder"]
            for i, col in enumerate(self.categorical_cols):
                for category in cat_encoder.categories_[i]:
                    feature_names.append(f"{col}_{category}")

        # Numerical features
        feature_names.extend(self.numerical_cols)

        self.feature_names_ = feature_names

    def transform(self, X):
        if self.preprocessor is None:
            raise ValueError("Not fitted yet")
        return self.preprocessor.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names(self):
        if self.feature_names_ is None:
            raise ValueError("Not fitted yet")
        return self.feature_names_

    def get_feature_importance_mapping(self):
        """Map transformed feature indices back to original column names.
        Useful for aggregating SHAP values across one-hot encoded features."""
        mapping = {}

        if self.categorical_cols:
            cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["encoder"]
            idx = 0
            for i, col in enumerate(self.categorical_cols):
                for _ in cat_encoder.categories_[i]:
                    mapping[idx] = col
                    idx += 1

        for col in self.numerical_cols:
            mapping[len(mapping)] = col

        return mapping


if __name__ == "__main__":
    from loader import CreditDataLoader

    loader = CreditDataLoader()
    X_train, X_test, y_train, y_test = loader.get_train_test_split()

    preprocessor = CreditDataPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Original shape: {X_train.shape}")
    print(f"Processed shape: {X_train_processed.shape}")
    print(f"Feature names: {len(preprocessor.get_feature_names())}")
