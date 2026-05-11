from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from statistics import mean
from typing import Any


@dataclass
class ModelResult:
    name: str
    problem_type: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    model: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def train_models(x_train: list[list[float]], y_train: list[Any], x_eval: list[list[float]], y_eval: list[Any], problem_type: str) -> list[ModelResult]:
    if problem_type == "regression":
        candidates = [_train_mean(y_train), _train_knn(x_train, y_train, 3, problem_type), _train_knn(x_train, y_train, 5, problem_type)]
    else:
        candidates = [_train_majority(y_train), _train_knn(x_train, y_train, 3, problem_type), _train_knn(x_train, y_train, 5, problem_type)]
    for result in candidates:
        preds = predict(result, x_eval)
        result.metrics = evaluate(y_eval, preds, problem_type)
    return sorted(candidates, key=lambda r: _score(r), reverse=True)


def tune_knn(x_train: list[list[float]], y_train: list[Any], x_eval: list[list[float]], y_eval: list[Any], problem_type: str) -> ModelResult:
    tuned = [_train_knn(x_train, y_train, k, problem_type) for k in (1, 3, 5, 7)]
    for result in tuned:
        result.metrics = evaluate(y_eval, predict(result, x_eval), problem_type)
    return sorted(tuned, key=lambda r: _score(r), reverse=True)[0]


def _train_majority(y: list[Any]) -> ModelResult:
    label = Counter(map(str, y)).most_common(1)[0][0]
    return ModelResult("Majority Classifier", "classification", {}, {}, {"label": label})


def _train_mean(y: list[Any]) -> ModelResult:
    return ModelResult("Mean Regressor", "regression", {}, {}, {"value": mean(float(v) for v in y)})


def _train_knn(x: list[list[float]], y: list[Any], k: int, problem_type: str) -> ModelResult:
    return ModelResult(f"KNN k={k}", problem_type, {"k": k}, {}, {"x": x, "y": y})


def predict(result: ModelResult, x_rows: list[list[float]]) -> list[Any]:
    if result.name == "Majority Classifier":
        return [result.model["label"] for _ in x_rows]
    if result.name == "Mean Regressor":
        return [result.model["value"] for _ in x_rows]
    k = int(result.parameters["k"])
    train_x = result.model["x"]
    train_y = result.model["y"]
    predictions = []
    for row in x_rows:
        neighbors = sorted(zip(train_x, train_y), key=lambda pair: _distance(row, pair[0]))[:k]
        labels = [label for _, label in neighbors]
        if result.problem_type == "regression":
            predictions.append(mean(float(v) for v in labels))
        else:
            predictions.append(Counter(map(str, labels)).most_common(1)[0][0])
    return predictions


def evaluate(actual: list[Any], predicted: list[Any], problem_type: str) -> dict[str, float]:
    if not actual:
        return {}
    if problem_type == "regression":
        errors = [float(a) - float(p) for a, p in zip(actual, predicted)]
        mae = mean(abs(e) for e in errors)
        rmse = math.sqrt(mean(e * e for e in errors))
        baseline = mean(float(a) for a in actual)
        ss_tot = sum((float(a) - baseline) ** 2 for a in actual) or 1.0
        ss_res = sum(e * e for e in errors)
        return {"mae": mae, "rmse": rmse, "r2": 1 - ss_res / ss_tot}
    actual_s = list(map(str, actual))
    predicted_s = list(map(str, predicted))
    correct = sum(a == p for a, p in zip(actual_s, predicted_s))
    labels = sorted(set(actual_s) | set(predicted_s))
    per_label = []
    for label in labels:
        tp = sum(a == p == label for a, p in zip(actual_s, predicted_s))
        fp = sum(a != label and p == label for a, p in zip(actual_s, predicted_s))
        fn = sum(a == label and p != label for a, p in zip(actual_s, predicted_s))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label.append((precision, recall, f1))
    return {
        "accuracy": correct / len(actual_s),
        "precision": mean(v[0] for v in per_label),
        "recall": mean(v[1] for v in per_label),
        "f1": mean(v[2] for v in per_label),
    }


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def _score(result: ModelResult) -> float:
    return result.metrics.get("r2", result.metrics.get("f1", result.metrics.get("accuracy", -999)))


def model_to_json(result: ModelResult) -> str:
    return json.dumps(result.to_json(), indent=2, sort_keys=True)
