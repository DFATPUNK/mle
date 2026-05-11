from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .data import infer_type

Row = dict[str, Any]


@dataclass
class FeaturePipeline:
    target: str
    problem_type: str
    numeric_features: list[str]
    categorical_features: list[str]
    categories: dict[str, list[str]]
    means: dict[str, float]
    leakage_warnings: list[str]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def detect_problem(rows: list[Row], target: str) -> str:
    values = [r.get(target) for r in rows if r.get(target) is not None]
    target_type = infer_type(values)
    if target_type in {"integer", "number"} and len(set(values)) > 10:
        return "regression"
    return "classification"


def build_feature_pipeline(rows: list[Row], target: str) -> FeaturePipeline:
    if not rows or target not in rows[0]:
        raise ValueError("Target column must exist in the dataset")
    problem = detect_problem(rows, target)
    candidates = [c for c in rows[0] if c != target]
    types = {c: infer_type([r.get(c) for r in rows]) for c in candidates}
    numeric = [c for c, t in types.items() if t in {"integer", "number", "boolean"}]
    categorical = [c for c, t in types.items() if t in {"category", "text"}]
    categories = {c: sorted({str(r.get(c)) for r in rows}) for c in categorical}
    means = {c: sum(float(r.get(c) or 0) for r in rows) / len(rows) for c in numeric}
    warnings = [c for c in candidates if target.lower() in c.lower() or c.lower() in {"label", "outcome"}]
    return FeaturePipeline(target, problem, numeric, categorical, categories, means, warnings)


def transform_rows(rows: list[Row], fp: FeaturePipeline) -> tuple[list[list[float]], list[Any]]:
    matrix: list[list[float]] = []
    labels: list[Any] = []
    for row in rows:
        vector: list[float] = []
        for col in fp.numeric_features:
            value = row.get(col, fp.means[col])
            try:
                vector.append(float(value))
            except (TypeError, ValueError):
                vector.append(fp.means[col])
        for col in fp.categorical_features:
            value = str(row.get(col))
            vector.extend(1.0 if value == cat else 0.0 for cat in fp.categories[col])
        matrix.append(vector)
        labels.append(row.get(fp.target))
    return matrix, labels
