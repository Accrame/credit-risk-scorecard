#!/usr/bin/env python3
"""
Credit Risk Scorecard - Main Pipeline
Run the complete credit risk modeling workflow.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def run_pipeline(
    data_path: str = None,
    output_dir: str = "models",
    run_fairness: bool = True,
):
    """
    Run the complete credit risk modeling pipeline.
    
    Steps:
    1. Load and preprocess data
    2. Engineer features
    3. Train model
    4. Evaluate performance
    5. Generate explanations
    6. Audit fairness
    7. Save artifacts
    """
    from src.data.loader import CreditDataLoader
    from src.data.preprocessor import CreditDataPreprocessor
    from src.features.engineer import CreditFeatureEngineer
    from src.models.trainer import CreditRiskTrainer, print_evaluation_report
    from src.models.scorecard import ScorecardConverter
    from src.explainability.shap_explainer import CreditRiskExplainer
    from src.explainability.fairness import FairnessAuditor, create_age_groups, extract_gender_from_personal_status
    
    print("=" * 60)
    print("CREDIT RISK SCORECARD PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1/7] Loading data...")
    loader = CreditDataLoader(data_dir="data")
    
    if data_path:
        df = pd.read_csv(data_path)
    else:
        df = loader.fetch_data()
    
    print(f"  Loaded {len(df)} records with {len(df.columns)} columns")
    print(f"  Default rate: {df['target'].mean():.2%}")
    
    # Step 2: Feature engineering
    print("\n[2/7] Engineering features...")
    X = df.drop("target", axis=1)
    y = df["target"]
    
    engineer = CreditFeatureEngineer()
    X_engineered = engineer.fit_transform(X)
    print(f"  Original features: {len(X.columns)}")
    print(f"  After engineering: {len(X_engineered.columns)}")
    
    # Keep original data for fairness analysis
    X_original = X.copy()
    
    # Step 3: Preprocessing
    print("\n[3/7] Preprocessing...")
    preprocessor = CreditDataPreprocessor()
    
    # Split data first
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_orig, X_test_orig = train_test_split(
        X_original, test_size=0.2, random_state=42, stratify=y
    )[:2]
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    feature_names = preprocessor.get_feature_names()
    print(f"  Processed features: {X_train_processed.shape[1]}")
    
    # Step 4: Train model
    print("\n[4/7] Training model...")
    trainer = CreditRiskTrainer()
    trainer.train(
        X_train_processed, y_train.values,
        X_val=X_test_processed, y_val=y_test.values,
        feature_names=feature_names,
    )
    
    # Cross-validation
    cv_results = trainer.cross_validate(X_train_processed, y_train.values)
    print(f"  CV AUC: {cv_results['cv_auc_mean']:.4f} (+/- {cv_results['cv_auc_std']:.4f})")
    
    # Step 5: Evaluate
    print("\n[5/7] Evaluating model...")
    metrics = trainer.evaluate(X_test_processed, y_test.values)
    print_evaluation_report(metrics)
    
    # Step 6: Explanations
    print("\n[6/7] Generating explanations...")
    explainer = CreditRiskExplainer(trainer.model, feature_names)
    explainer.fit(X_train_processed)
    
    # Global feature importance
    importance = explainer.get_global_importance(X_test_processed)
    print("\n  Top 10 Important Features:")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance_pct']:.1f}%")
    
    # Example prediction explanation
    sample_idx = 0
    explanation = explainer.explain_prediction(X_test_processed[sample_idx])
    print(f"\n  Sample prediction: {explanation['final_probability']:.2%} default probability")
    
    # Scorecard conversion
    converter = ScorecardConverter()
    y_proba = trainer.predict_proba(X_test_processed)
    scores = converter.probability_to_score(y_proba)
    print(f"\n  Score range: {scores.min()} - {scores.max()}")
    print(f"  Mean score: {scores.mean():.0f}")
    
    # Step 7: Fairness audit
    if run_fairness:
        print("\n[7/7] Auditing fairness...")
        
        # Create sensitive features
        sensitive_data = pd.DataFrame()
        
        if "age" in X_test_orig.columns:
            sensitive_data["age_group"] = create_age_groups(X_test_orig["age"].values)
        
        if "personal_status_sex" in X_test_orig.columns:
            sensitive_data["gender"] = extract_gender_from_personal_status(
                X_test_orig["personal_status_sex"]
            )

        # Drop rows with missing values in sensitive features
        sensitive_data = sensitive_data.fillna("unknown")

        if len(sensitive_data.columns) > 0:
            auditor = FairnessAuditor(list(sensitive_data.columns))
            y_pred = trainer.predict(X_test_processed)
            fairness_results = auditor.audit(y_test.values, y_pred, sensitive_data)
            
            print("\n  Fairness Results:")
            for feature, results in fairness_results.items():
                dp = results["demographic_parity_difference"]
                ratio = results["demographic_parity_ratio"]
                print(f"\n  {feature}:")
                print(f"    Demographic Parity Difference: {dp:.3f}")
                print(f"    Demographic Parity Ratio: {ratio:.3f}")
                
                compliance = results["compliance"]
                for check, passed in compliance.items():
                    status = "PASS" if passed else "FAIL"
                    print(f"    {check}: {status}")
    
    # Save artifacts
    print("\n" + "=" * 60)
    print("Saving artifacts...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    trainer.save(output_path / "credit_risk_model.joblib")
    importance.to_csv(output_path / "feature_importance.csv", index=False)
    
    # Save score distribution
    score_analysis = converter.analyze_score_distribution(scores, y_test.values)
    score_analysis.to_csv(output_path / "score_distribution.csv", index=False)
    
    print(f"  Model saved to: {output_path / 'credit_risk_model.joblib'}")
    print(f"  Feature importance saved to: {output_path / 'feature_importance.csv'}")
    print(f"  Score distribution saved to: {output_path / 'score_distribution.csv'}")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    return {
        "metrics": metrics,
        "cv_results": cv_results,
        "feature_importance": importance,
    }


def main():
    parser = argparse.ArgumentParser(description="Credit Risk Scorecard Pipeline")
    parser.add_argument("--data", type=str, help="Path to input data CSV")
    parser.add_argument("--output", type=str, default="models", help="Output directory")
    parser.add_argument("--no-fairness", action="store_true", help="Skip fairness audit")
    
    args = parser.parse_args()
    
    run_pipeline(
        data_path=args.data,
        output_dir=args.output,
        run_fairness=not args.no_fairness,
    )


if __name__ == "__main__":
    main()
