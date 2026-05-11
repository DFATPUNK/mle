from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

Row = dict[str, Any]


def normalize_column(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")
    return normalized or "column"


def _coerce(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    return value


def load_csv(path: str | Path) -> list[Row]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = [normalize_column(h or "column") for h in (reader.fieldnames or [])]
        rows: list[Row] = []
        for raw in reader:
            row: Row = {}
            for original, normalized in zip(reader.fieldnames or [], headers):
                row[normalized] = _coerce(raw.get(original, ""))
            rows.append(row)
        return rows


def infer_type(values: list[Any]) -> str:
    present = [v for v in values if v is not None]
    if not present:
        return "unknown"
    if all(isinstance(v, bool) for v in present):
        return "boolean"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in present):
        return "integer"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
        return "number"
    return "category" if len(set(map(str, present))) <= max(20, len(present) // 2) else "text"


@dataclass
class DatasetProfile:
    row_count: int
    column_count: int
    columns: dict[str, str]
    missing_values: dict[str, int]
    duplicate_rows: int
    fingerprint: str


def profile_dataset(rows: list[Row]) -> DatasetProfile:
    columns = list(rows[0].keys()) if rows else []
    duplicate_rows = len(rows) - len({json.dumps(r, sort_keys=True, default=str) for r in rows})
    payload = json.dumps(rows, sort_keys=True, default=str).encode("utf-8")
    return DatasetProfile(
        row_count=len(rows),
        column_count=len(columns),
        columns={c: infer_type([r.get(c) for r in rows]) for c in columns},
        missing_values={c: sum(1 for r in rows if r.get(c) is None) for c in columns},
        duplicate_rows=duplicate_rows,
        fingerprint=hashlib.sha256(payload).hexdigest()[:16],
    )


def clean_rows(rows: list[Row]) -> tuple[list[Row], list[str]]:
    if not rows:
        return [], ["Dataset is empty"]
    log: list[str] = []
    seen: set[str] = set()
    deduped: list[Row] = []
    for row in rows:
        key = json.dumps(row, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            deduped.append(dict(row))
    if len(deduped) != len(rows):
        log.append(f"Removed {len(rows) - len(deduped)} duplicate rows")

    columns = list(deduped[0])
    types = {c: infer_type([r.get(c) for r in deduped]) for c in columns}
    for column in columns:
        present = [r[column] for r in deduped if r.get(column) is not None]
        if not present:
            fill = 0
        elif types[column] in {"integer", "number"}:
            fill = sum(float(v) for v in present) / len(present)
        else:
            fill = max(set(map(str, present)), key=list(map(str, present)).count)
        missing = 0
        for row in deduped:
            if row.get(column) is None:
                row[column] = fill
                missing += 1
        if missing:
            log.append(f"Imputed {missing} missing values in {column}")
    return deduped, log


@dataclass
class DatasetVersion:
    name: str
    version: int
    created_at: str
    source: str
    profile: DatasetProfile
    transformations: list[str]
    rows: list[Row]

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["profile"] = asdict(self.profile)
        return data


def make_version(name: str, source: str, rows: list[Row], transformations: list[str], version: int = 1) -> DatasetVersion:
    return DatasetVersion(
        name=name,
        version=version,
        created_at=datetime.now(timezone.utc).isoformat(),
        source=source,
        profile=profile_dataset(rows),
        transformations=transformations,
        rows=rows,
    )
