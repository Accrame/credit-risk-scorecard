"""Tests for the model training and scoring modules."""

import numpy as np
import pytest
from sklearn.datasets import make_classification


class TestCreditRiskTrainer:
    """Tests for the CreditRiskTrainer class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data."""
        X, y = make_classification(
            n_samples=500,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            random_state=42,
        )
        return X, y

    @pytest.fixture
    def train_test_data(self, sample_data):
        """Split sample data into train/test."""
        from sklearn.model_selection import train_test_split

        X, y = sample_data
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def test_trainer_initialization(self):
        """Test trainer initializes correctly."""
        from src.models.trainer import CreditRiskTrainer

        trainer = CreditRiskTrainer(model_type="xgboost")
        assert trainer.model_type == "xgboost"
        assert trainer.model is None

    def test_trainer_train(self, train_test_data):
        """Test model training."""
        from src.models.trainer import CreditRiskTrainer

        X_train, X_test, y_train, y_test = train_test_data

        trainer = CreditRiskTrainer(model_type="xgboost")
        trainer.train(X_train, y_train)

        assert trainer.model is not None

    def test_trainer_predict(self, train_test_data):
        """Test model prediction."""
        from src.models.trainer import CreditRiskTrainer

        X_train, X_test, y_train, y_test = train_test_data

        trainer = CreditRiskTrainer(model_type="xgboost")
        trainer.train(X_train, y_train)

        predictions = trainer.predict(X_test)

        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})

    def test_trainer_predict_proba(self, train_test_data):
        """Test probability prediction."""
        from src.models.trainer import CreditRiskTrainer

        X_train, X_test, y_train, y_test = train_test_data

        trainer = CreditRiskTrainer(model_type="xgboost")
        trainer.train(X_train, y_train)

        probas = trainer.predict_proba(X_test)

        assert len(probas) == len(y_test)
        assert all(0 <= p <= 1 for p in probas)

    def test_trainer_evaluate(self, train_test_data):
        """Test model evaluation."""
        from src.models.trainer import CreditRiskTrainer

        X_train, X_test, y_train, y_test = train_test_data

        trainer = CreditRiskTrainer(model_type="xgboost")
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)

        assert "auc_roc" in metrics
        assert "gini" in metrics
        assert "ks_statistic" in metrics
        assert 0 <= metrics["auc_roc"] <= 1

    def test_trainer_cross_validate(self, sample_data):
        """Test cross-validation."""
        from src.models.trainer import CreditRiskTrainer

        X, y = sample_data

        trainer = CreditRiskTrainer(model_type="xgboost")
        cv_results = trainer.cross_validate(X, y, n_splits=3)

        assert "cv_auc_mean" in cv_results
        assert "cv_auc_std" in cv_results
        assert len(cv_results["cv_scores"]) == 3


class TestScorecardConverter:
    """Tests for the ScorecardConverter class."""

    def test_probability_to_score(self):
        """Test probability to score conversion."""
        from src.models.scorecard import ScorecardConverter

        converter = ScorecardConverter()

        # Low probability -> high score
        low_prob_score = converter.probability_to_score(np.array([0.05]))[0]
        # High probability -> low score
        high_prob_score = converter.probability_to_score(np.array([0.50]))[0]

        assert low_prob_score > high_prob_score
        assert 300 <= low_prob_score <= 850
        assert 300 <= high_prob_score <= 850

    def test_score_to_probability_inverse(self):
        """Test score to probability is inverse of probability to score."""
        from src.models.scorecard import ScorecardConverter

        converter = ScorecardConverter()

        original_prob = np.array([0.15])
        score = converter.probability_to_score(original_prob)
        recovered_prob = converter.score_to_probability(score)

        np.testing.assert_almost_equal(original_prob, recovered_prob, decimal=2)

    def test_score_bands(self):
        """Test score band generation."""
        from src.models.scorecard import ScorecardConverter

        converter = ScorecardConverter()
        bands = converter.get_score_bands()

        assert len(bands) > 0
        assert "min_score" in bands.columns
        assert "max_score" in bands.columns
        assert "risk_level" in bands.columns


class TestDataLoader:
    """Tests for the CreditDataLoader class."""

    def test_loader_column_names(self):
        """Test that column names are defined correctly."""
        from src.data.loader import COLUMN_NAMES

        assert len(COLUMN_NAMES) == 21
        assert "credit_amount" in COLUMN_NAMES
        assert "credit_risk" in COLUMN_NAMES


class TestPreprocessor:
    """Tests for the CreditDataPreprocessor class."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        import pandas as pd

        return pd.DataFrame(
            {
                "num1": [1, 2, 3, 4, 5],
                "num2": [10, 20, 30, 40, 50],
                "cat1": ["a", "b", "a", "b", "c"],
                "cat2": ["x", "x", "y", "y", "y"],
            }
        )

    def test_preprocessor_fit_transform(self, sample_df):
        """Test preprocessor fit and transform."""
        from src.data.preprocessor import CreditDataPreprocessor

        preprocessor = CreditDataPreprocessor()
        X_transformed = preprocessor.fit_transform(sample_df)

        assert X_transformed is not None
        assert len(X_transformed) == len(sample_df)

    def test_preprocessor_feature_names(self, sample_df):
        """Test feature name extraction."""
        from src.data.preprocessor import CreditDataPreprocessor

        preprocessor = CreditDataPreprocessor()
        preprocessor.fit_transform(sample_df)

        feature_names = preprocessor.get_feature_names()

        assert len(feature_names) > 0
        # Should have more features due to one-hot encoding
        assert len(feature_names) >= len(sample_df.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
