from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

Row = dict[str, Any]


def split_dataset(rows: list[Row], train: float = 0.7, test: float = 0.15, validation: float = 0.15,
                  mode: str = "random", target: str | None = None, seed: int = 42) -> dict[str, list[Row]]:
    if round(train + test + validation, 6) != 1:
        raise ValueError("Split proportions must add up to 1.0")
    rng = random.Random(seed)
    ordered = list(rows)
    if mode == "stratified" and target:
        buckets: dict[Any, list[Row]] = defaultdict(list)
        for row in ordered:
            buckets[row.get(target)].append(row)
        parts = {"train": [], "test": [], "validation": []}
        for bucket in buckets.values():
            rng.shuffle(bucket)
            _assign(bucket, train, test, parts)
        return parts
    if mode == "random":
        rng.shuffle(ordered)
    elif mode != "time-based":
        raise ValueError(f"Unsupported split mode: {mode}")
    parts = {"train": [], "test": [], "validation": []}
    _assign(ordered, train, test, parts)
    return parts


def _assign(rows: list[Row], train: float, test: float, parts: dict[str, list[Row]]) -> None:
    n = len(rows)
    train_end = int(n * train)
    test_end = train_end + int(n * test)
    parts["train"].extend(rows[:train_end])
    parts["test"].extend(rows[train_end:test_end])
    parts["validation"].extend(rows[test_end:])
