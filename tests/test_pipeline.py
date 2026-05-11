from pathlib import Path

from mle_no_code import NoCodeMLPipeline
from mle_no_code.data import normalize_column, profile_dataset


def write_sample_csv(path: Path) -> None:
    path.write_text(
        "Age,Plan,Visits,Churn\n"
        "20,basic,3,no\n"
        "21,basic,2,no\n"
        "45,pro,10,yes\n"
        "44,pro,9,yes\n"
        "23,basic,,no\n"
        "50,enterprise,11,yes\n"
        "50,enterprise,11,yes\n",
        encoding="utf-8",
    )


def test_profile_detects_schema_and_duplicates() -> None:
    rows = [{"a": 1, "b": "x"}, {"a": 1, "b": "x"}, {"a": None, "b": "y"}]
    profile = profile_dataset(rows)
    assert profile.row_count == 3
    assert profile.duplicate_rows == 1
    assert profile.columns["a"] == "integer"
    assert profile.missing_values["a"] == 1
    assert normalize_column("Customer ID!") == "customer_id"


def test_end_to_end_classification_pipeline(tmp_path: Path) -> None:
    csv_path = tmp_path / "customers.csv"
    write_sample_csv(csv_path)
    pipeline = NoCodeMLPipeline(tmp_path / "artifacts")

    dataset = pipeline.upload_csv(csv_path)
    assert dataset.profile.duplicate_rows == 0
    assert "visits" in dataset.profile.columns

    splits = pipeline.split(0.6, 0.2, 0.2, target="churn")
    assert set(splits) == {"train", "test", "validation"}

    features = pipeline.select_target("churn")
    assert features.problem_type == "classification"
    assert "plan" in features.categorical_features

    models = pipeline.train()
    tuned = pipeline.tune()
    artifact = pipeline.export_best_model()

    assert models
    assert tuned.name.startswith("KNN")
    assert artifact.exists()

    prediction = pipeline.predict_one({"age": 46, "plan": "pro", "visits": 8})
    assert prediction["status"] == "success"
    feedback = pipeline.add_feedback(0, "yes", user="qa")
    assert feedback["corrected_value"] == "yes"
    assert pipeline.workflow_blocks()[0]["name"] == "Data Source"
