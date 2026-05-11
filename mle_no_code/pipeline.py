from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data import DatasetVersion, Row, clean_rows, load_csv, make_version
from .features import FeaturePipeline, build_feature_pipeline, transform_rows
from .models import ModelResult, predict, train_models, tune_knn
from .split import split_dataset


@dataclass
class BlockLog:
    block: str
    status: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class NoCodeMLPipeline:
    """Executable backend for visual blocks in the no-code ML builder."""

    def __init__(self, artifact_dir: str | Path = "artifacts") -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.logs: list[BlockLog] = []
        self.dataset: DatasetVersion | None = None
        self.splits: dict[str, list[Row]] = {}
        self.feature_pipeline: FeaturePipeline | None = None
        self.models: list[ModelResult] = []
        self.best_model: ModelResult | None = None
        self.prediction_logs: list[dict[str, Any]] = []
        self.feedback: list[dict[str, Any]] = []

    def upload_csv(self, path: str | Path, name: str | None = None) -> DatasetVersion:
        rows = load_csv(path)
        self._log("Data Source", "success", f"Loaded {len(rows)} rows from CSV")
        cleaned, transformations = clean_rows(rows)
        self._log("ETL", "success", "; ".join(transformations) or "No cleaning required")
        self.dataset = make_version(name or Path(path).stem, str(path), cleaned, transformations)
        self._save_json(f"dataset_{self.dataset.name}_v{self.dataset.version}.json", self.dataset.to_json())
        return self.dataset

    def split(self, train: float = 0.7, test: float = 0.15, validation: float = 0.15, mode: str = "random", target: str | None = None) -> dict[str, list[Row]]:
        if self.dataset is None:
            raise RuntimeError("Upload a dataset before splitting")
        self.splits = split_dataset(self.dataset.rows, train, test, validation, mode, target)
        self._log("Split Data", "success", f"Created train/test/validation splits: { {k: len(v) for k, v in self.splits.items()} }")
        return self.splits

    def select_target(self, target: str) -> FeaturePipeline:
        train_rows = self.splits.get("train") or (self.dataset.rows if self.dataset else [])
        self.feature_pipeline = build_feature_pipeline(train_rows, target)
        problem = self.feature_pipeline.problem_type
        self._log("Training Set", "success", f"Target '{target}' selected; detected {problem}")
        if self.feature_pipeline.leakage_warnings:
            self._log("Training Set", "warning", f"Possible leakage columns: {self.feature_pipeline.leakage_warnings}")
        self._save_json("feature_pipeline.json", self.feature_pipeline.to_json())
        return self.feature_pipeline

    def train(self) -> list[ModelResult]:
        if not self.feature_pipeline or not self.splits:
            raise RuntimeError("Split data and select a target before training")
        x_train, y_train = transform_rows(self.splits["train"], self.feature_pipeline)
        eval_rows = self.splits.get("test") or self.splits.get("validation") or self.splits["train"]
        x_eval, y_eval = transform_rows(eval_rows, self.feature_pipeline)
        self.models = train_models(x_train, y_train, x_eval, y_eval, self.feature_pipeline.problem_type)
        self.best_model = self.models[0]
        self._log("Train Model", "success", f"Trained {len(self.models)} models; best is {self.best_model.name}")
        self._save_json("model_evaluations.json", [m.to_json() for m in self.models])
        return self.models

    def tune(self) -> ModelResult:
        if not self.feature_pipeline or not self.splits:
            raise RuntimeError("Train prerequisites are missing")
        x_train, y_train = transform_rows(self.splits["train"], self.feature_pipeline)
        x_eval, y_eval = transform_rows(self.splits.get("validation") or self.splits["test"], self.feature_pipeline)
        tuned = tune_knn(x_train, y_train, x_eval, y_eval, self.feature_pipeline.problem_type)
        self.models.append(tuned)
        self.best_model = sorted(self.models, key=lambda m: m.metrics.get("r2", m.metrics.get("f1", m.metrics.get("accuracy", -999))), reverse=True)[0]
        self._log("Tune", "success", f"Best tuned candidate: {tuned.name} with {tuned.metrics}")
        return tuned

    def export_best_model(self, filename: str = "best_model.json") -> Path:
        if not self.best_model or not self.feature_pipeline or not self.dataset:
            raise RuntimeError("No trained model is available to export")
        artifact = {
            "model": self.best_model.to_json(),
            "preprocessing": self.feature_pipeline.to_json(),
            "input_schema": self.dataset.profile.columns,
            "dataset_version": self.dataset.version,
            "metrics": self.best_model.metrics,
            "training_config": self.best_model.parameters,
            "generated_documentation": self.documentation(),
        }
        return self._save_json(filename, artifact)

    def predict_one(self, row: Row) -> dict[str, Any]:
        if not self.best_model or not self.feature_pipeline:
            raise RuntimeError("Train or load a model before prediction")
        x, _ = transform_rows([row | {self.feature_pipeline.target: None}], self.feature_pipeline)
        output = predict(self.best_model, x)[0]
        record = {
            "input": row,
            "output": output,
            "model": self.best_model.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "score_confidence": self.best_model.metrics.get("accuracy", self.best_model.metrics.get("r2", 0.0)),
            "status": "success",
        }
        self.prediction_logs.append(record)
        self._save_json("prediction_logs.json", self.prediction_logs)
        return record

    def add_feedback(self, prediction_index: int, corrected_value: Any, user: str = "manual") -> dict[str, Any]:
        item = {"prediction_index": prediction_index, "corrected_value": corrected_value, "user": user, "timestamp": datetime.now(timezone.utc).isoformat()}
        self.feedback.append(item)
        self._save_json("feedback_queue.json", self.feedback)
        self._log("Feedback", "success", "Stored manual correction for retraining")
        return item

    def retrain(self) -> ModelResult:
        self.train()
        return self.best_model  # type: ignore[return-value]

    def documentation(self) -> str:
        if not self.best_model or not self.feature_pipeline:
            return "Pipeline not trained yet."
        return (
            f"No-code ML pipeline using target '{self.feature_pipeline.target}' for "
            f"{self.feature_pipeline.problem_type}. Best model: {self.best_model.name}. "
            f"Metrics: {self.best_model.metrics}."
        )

    def workflow_blocks(self) -> list[dict[str, Any]]:
        names = ["Data Source", "ETL", "Data Warehouse", "Split Data", "Feature Engineering", "Train Model", "Evaluate", "Tune", "Deploy", "Monitor", "Feedback", "Retrain"]
        return [{"name": name, "inputs": [], "outputs": [], "status": self._status_for(name), "retry": True, "logs": [l.__dict__ for l in self.logs if l.block == name]} for name in names]

    def _status_for(self, block: str) -> str:
        matches = [l for l in self.logs if l.block == block]
        return matches[-1].status if matches else "pending"

    def _log(self, block: str, status: str, message: str) -> None:
        self.logs.append(BlockLog(block, status, message))

    def _save_json(self, filename: str, payload: Any) -> Path:
        path = self.artifact_dir / filename
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return path
