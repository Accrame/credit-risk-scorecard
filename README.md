# Credit Risk Scorecard

Credit risk scoring model using XGBoost on the German Credit Dataset, with SHAP explanations and fairness checks.

## What this does

Predicts loan default probability and converts it to a traditional credit score (300-850 range). The model also generates adverse action reasons — basically telling the applicant *why* they were rejected, which banks are legally required to do (ECOA, GDPR Article 22).

## Dataset

[German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) from UCI — 1000 loan applications, 20 features, binary outcome. Not huge, but it's a standard benchmark and the categorical encoding is a pain to deal with (everything is coded as A11, A12, etc).

## Results

| Metric | Score |
|--------|-------|
| AUC-ROC | 0.78 |
| Gini | 0.56 |
| KS | 0.46 |

Not amazing, but reasonable for 1000 samples with no heavy tuning.

## Setup

```bash
pip install -r requirements.txt
python main.py
```

Or run the dashboard:
```bash
streamlit run streamlit_app/app.py
```

## Project structure

- `src/data/` — loading + preprocessing the German Credit data
- `src/features/` — feature engineering (debt ratios, stability scores, etc)
- `src/models/` — XGBoost trainer + scorecard conversion
- `src/explainability/` — SHAP explanations + Fairlearn fairness audit
- `streamlit_app/` — interactive dashboard
- `tests/` — unit tests

## Explainability

Each prediction comes with SHAP values showing which features pushed the score up or down. There's also an adverse action module that maps SHAP contributions to human-readable denial reasons (e.g. "Insufficient checking account history").

## Fairness

The model gets audited across age groups and gender using Fairlearn. Checks the four-fifths rule (80% rule) and demographic parity. On this dataset the model passes, but barely — the age group disparity is close to the threshold.

## What I'd do differently

- The German Credit dataset is from 1994 and uses Deutsche Marks. Would be better with more recent data
- I should have tried WoE (Weight of Evidence) binning — that's what actual banks use for scorecards
- The feature engineering is manual. Could try automated feature selection with Boruta or similar
- Hyperparameter tuning is basically default XGBoost params with minor tweaks

## License

MIT
