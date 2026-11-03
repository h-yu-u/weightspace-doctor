"""Minimal JSON persistence; model internals belong elsewhere."""

import json
from pathlib import Path
from typing import Any

from .schema import Case, ContrastPair, Trace


def jsonl_read(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    items: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("JSONL record must be an object")
            items.append(obj)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return items


def unique_ids(items: list[Any]) -> None:
    ids = [item.id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate ids found; examples must have unique ids")


def read_cases(path: str | Path) -> list[Case]:
    cases = [Case.from_dict(d) for d in jsonl_read(path)]
    unique_ids(cases)
    return cases


def read_traces(path: str | Path) -> list[Trace]:
    traces = [Trace.from_dict(d) for d in jsonl_read(path)]
    unique_ids(traces)
    return traces


def read_pairs(path: str | Path) -> list[ContrastPair]:
    pairs = [ContrastPair.from_dict(d) for d in jsonl_read(path)]
    unique_ids(pairs)
    return pairs


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                 encoding="utf-8")


def write_jsonl(path: str | Path, records: list[Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")
