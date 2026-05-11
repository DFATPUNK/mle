# No-code Machine Learning Pipeline

This repository contains an MVP for a Zapier-inspired, no-code machine-learning pipeline builder. The goal is to guide a user from data connection through cleaning, splitting, feature engineering, model training, evaluation, tuning, deployment artifacts, prediction logs, feedback, and retraining without requiring code in the product experience.

## MVP scope implemented

The first version focuses on the recommended MVP flow:

```text
Upload CSV → Auto clean → Split Data → Select Target → Feature Engineering → Train 3 Models → Evaluate → Tune → Export → Predict → Feedback → Retrain
```

Implemented capabilities:

1. CSV upload with automatic column normalization.
2. Dataset profiling for columns, types, missing values, duplicates, row counts, and fingerprints.
3. Simple ETL cleaning with duplicate removal and missing-value imputation.
4. Dataset version metadata suitable for a lightweight warehouse layer.
5. Train/test/validation splitting with random, stratified, and time-based modes.
6. Target selection and automatic classification/regression detection.
7. Basic feature engineering with numeric passthrough and categorical one-hot encoding.
8. Training/evaluation of three simple standard-library models: majority/mean baseline plus KNN variants.
9. Simple hyperparameter tuning for KNN values.
10. Model export with preprocessing, schema, dataset version, metrics, and generated documentation.
11. Prediction logging with inputs, outputs, model, timestamp, confidence score, and status.
12. Manual feedback queue and retrain entry point.
13. A minimal HTTP interface that renders workflow blocks and exposes upload, status, predict, and feedback endpoints.

## Project layout

```text
mle_no_code/
  api.py       # stdlib HTTP no-code API and simple workflow page
  data.py      # CSV loading, profiling, cleaning, dataset versioning
  features.py  # problem detection and reusable preprocessing pipeline
  models.py    # simple baseline/KNN training, evaluation, tuning, prediction
  pipeline.py  # orchestration facade for visual blocks
  split.py     # train/test/validation splitting

tests/
  test_pipeline.py
```

## Run tests

```bash
python -m pytest
```

## Start the no-code API

```bash
python -m mle_no_code.api
```

Then open `http://127.0.0.1:8000`.

Example API calls:

```bash
curl -X POST 'http://127.0.0.1:8000/upload?path=/path/to/data.csv&target=churn'
curl -X GET 'http://127.0.0.1:8000/status'
curl -X POST 'http://127.0.0.1:8000/predict' \
  -H 'content-type: application/json' \
  -d '{"age": 46, "plan": "pro", "visits": 8}'
curl -X POST 'http://127.0.0.1:8000/feedback' \
  -H 'content-type: application/json' \
  -d '{"prediction_index": 0, "corrected_value": "yes", "user": "domain-expert"}'
```

## Next product milestones

- Add OAuth/API-key connectors for Google Sheets, Airtable, Notion, SQL databases, REST APIs, and data warehouses.
- Replace the minimal HTTP page with a full visual drag-and-drop workflow builder.
- Add persistent storage for datasets, artifacts, predictions, feedback, teams, permissions, and run histories.
- Add production-grade algorithms and integrations once dependencies such as pandas, scikit-learn, XGBoost, LightGBM, and FastAPI are introduced.
- Add monitoring dashboards for prediction volume, latency, API errors, input/output drift, performance drift, and alerts.
