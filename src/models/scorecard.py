"""Convert model probabilities to traditional credit scores (300-850 range).

Uses the standard Score = Offset + Factor * ln(odds) formula.
PDO (Points to Double Odds) controls how spread out the scores are.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ScorecardConverter:
    """Probability-to-score converter using log-odds scaling."""

    def __init__(self, target_score=600, target_odds=50, pdo=20, min_score=300, max_score=850):
        self.target_score = target_score
        self.target_odds = target_odds
        self.pdo = pdo
        self.min_score = min_score
        self.max_score = max_score

        # Calculate conversion parameters
        self.factor = pdo / np.log(2)
        self.offset = target_score - self.factor * np.log(target_odds)

    def probability_to_score(self, probability):
        # Clip probabilities to avoid log(0) or division by zero
        prob = np.clip(probability, 1e-10, 1 - 1e-10)

        # Calculate odds (good/bad)
        odds = (1 - prob) / prob

        # Convert to score
        score = self.offset + self.factor * np.log(odds)

        # Clip to valid range
        return np.clip(score, self.min_score, self.max_score).astype(int)

    def score_to_probability(self, score):
        """Inverse of probability_to_score."""
        # Reverse the formula
        log_odds = (score - self.offset) / self.factor
        odds = np.exp(log_odds)
        probability = 1 / (1 + odds)

        return probability

    def get_score_bands(self) -> pd.DataFrame:
        """
        Generate score bands with risk categorization.

        Returns:
            DataFrame with score bands and risk levels
        """
        bands = [
            (self.min_score, 499, "Very High Risk", "E"),
            (500, 549, "High Risk", "D"),
            (550, 599, "Medium-High Risk", "C"),
            (600, 649, "Medium Risk", "C+"),
            (650, 699, "Medium-Low Risk", "B"),
            (700, 749, "Low Risk", "B+"),
            (750, 799, "Very Low Risk", "A"),
            (800, self.max_score, "Excellent", "A+"),
        ]

        df = pd.DataFrame(bands, columns=["min_score", "max_score", "risk_level", "grade"])

        # Calculate probability range for each band
        df["max_pd"] = self.score_to_probability(df["min_score"])
        df["min_pd"] = self.score_to_probability(df["max_score"])

        return df

    def analyze_score_distribution(
        self,
        scores: np.ndarray,
        actuals: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Analyze the distribution of scores.

        Args:
            scores: Array of credit scores
            actuals: Actual default labels (optional)

        Returns:
            DataFrame with score distribution analysis
        """
        bands = self.get_score_bands()
        results = []

        for _, row in bands.iterrows():
            mask = (scores >= row["min_score"]) & (scores <= row["max_score"])
            count = mask.sum()
            pct = count / len(scores) * 100

            result = {
                "score_range": f"{row['min_score']}-{row['max_score']}",
                "grade": row["grade"],
                "risk_level": row["risk_level"],
                "count": count,
                "percentage": pct,
            }

            if actuals is not None and count > 0:
                result["actual_default_rate"] = actuals[mask].mean() * 100
                result["expected_pd_range"] = f"{row['min_pd']*100:.1f}%-{row['max_pd']*100:.1f}%"

            results.append(result)

        return pd.DataFrame(results)


class FeatureScorecard:
    """Points-based scorecard derived from SHAP values.
    Each feature/bin gets a point value, total score = base + sum of points."""

    def __init__(self, base_score=600, points_range=(-100, 100)):
        self.base_score = base_score
        self.points_range = points_range
        self.feature_points = {}

    def fit(self, shap_values, feature_names, X):
        # For each feature, create bins and calculate average SHAP contribution
        for i, feature in enumerate(feature_names):
            values = X.iloc[:, i] if isinstance(X, pd.DataFrame) else X[:, i]
            shap_contrib = shap_values[:, i]

            if pd.api.types.is_numeric_dtype(values):
                # Numerical: create bins
                try:
                    bins = pd.qcut(values, q=5, duplicates="drop")
                except ValueError:
                    bins = pd.cut(values, bins=5)

                points_list = []
                for bin_label in bins.unique():
                    mask = bins == bin_label
                    avg_shap = shap_contrib[mask].mean()
                    points = self._shap_to_points(avg_shap)
                    points_list.append((str(bin_label), points))

                self.feature_points[feature] = sorted(points_list, key=lambda x: x[1])
            else:
                # Categorical: group by value
                points_list = []
                for value in values.unique():
                    mask = values == value
                    if mask.sum() > 0:
                        avg_shap = shap_contrib[mask].mean()
                        points = self._shap_to_points(avg_shap)
                        points_list.append((str(value), points))

                self.feature_points[feature] = sorted(points_list, key=lambda x: x[1])

        return self

    def _shap_to_points(self, shap_value: float) -> int:
        """Convert SHAP value to points."""
        # Normalize SHAP to points range
        # Negative SHAP (reduces risk) -> positive points
        # Positive SHAP (increases risk) -> negative points
        normalized = -shap_value * 100  # Scale factor
        points = np.clip(normalized, self.points_range[0], self.points_range[1])
        return int(round(points))

    def get_scorecard_table(self) -> pd.DataFrame:
        """
        Get the full scorecard table.

        Returns:
            DataFrame with all features, bins, and points
        """
        rows = []
        for feature, points_list in self.feature_points.items():
            for bin_label, points in points_list:
                rows.append(
                    {
                        "feature": feature,
                        "value/range": bin_label,
                        "points": points,
                    }
                )

        return pd.DataFrame(rows)

    def score_applicant(self, features: dict[str, any]) -> dict:
        """
        Score a single applicant using the scorecard.

        Args:
            features: Dictionary of feature values

        Returns:
            Dictionary with score breakdown
        """
        total_points = self.base_score
        breakdown = []

        for feature, value in features.items():
            if feature in self.feature_points:
                # Find matching bin
                points = 0
                matched_bin = "unknown"

                for bin_label, bin_points in self.feature_points[feature]:
                    if str(value) == bin_label or str(value) in bin_label:
                        points = bin_points
                        matched_bin = bin_label
                        break

                total_points += points
                breakdown.append(
                    {
                        "feature": feature,
                        "value": value,
                        "matched_bin": matched_bin,
                        "points": points,
                    }
                )

        return {
            "total_score": total_points,
            "base_score": self.base_score,
            "breakdown": breakdown,
        }


if __name__ == "__main__":
    # Quick test
    converter = ScorecardConverter()

    # Test probabilities
    probs = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 0.70])
    scores = converter.probability_to_score(probs)

    print("Probability to Score conversion:")
    for p, s in zip(probs, scores):
        print(f"  PD: {p:.0%} -> Score: {s}")

    print("\nScore Bands:")
    print(converter.get_score_bands().to_string(index=False))
