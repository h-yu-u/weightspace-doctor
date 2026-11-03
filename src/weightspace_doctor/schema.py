"""Typed, transport-agnostic records. No I/O or framework dependencies."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Case:
    id: str
    prompt: str
    expected: str | None = None
    expected_regex: str | None = None
    split: str = "target"
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Case":
        if not str(d.get("id", "")).strip() or not str(d.get("prompt", "")).strip():
            raise ValueError("Each case requires nonempty id and prompt")
        if d.get("split", "target") not in ("target", "guard"):
            raise ValueError("split must be target or guard")
        return cls(id=str(d["id"]), prompt=str(d["prompt"]),
                   expected=d.get("expected"), expected_regex=d.get("expected_regex"),
                   split=d.get("split", "target"), tags=tuple(d.get("tags", [])))


@dataclass(frozen=True)
class Trace:
    id: str
    prompt: str
    text: str
    finish_reason: str = "unknown"
    output_tokens: int | None = None
    max_output_tokens: int | None = None
    elapsed_ms: int | None = None
    expected: str | None = None
    expected_regex: str | None = None
    split: str = "target"
    tags: tuple[str, ...] = ()
    error: str | None = None
    final_text: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trace":
        return cls(id=str(d["id"]), prompt=str(d.get("prompt", "")),
                   text=str(d.get("text", "")),
                   finish_reason=str(d.get("finish_reason", "unknown")),
                   output_tokens=d.get("output_tokens"),
                   max_output_tokens=d.get("max_output_tokens"),
                   elapsed_ms=d.get("elapsed_ms"), expected=d.get("expected"),
                   expected_regex=d.get("expected_regex"),
                   split=d.get("split", "target"), tags=tuple(d.get("tags", [])),
                   error=d.get("error"), final_text=d.get("final_text"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Finding:
    id: str
    split: str
    flags: tuple[str, ...]
    repetition_ratio: float
    correctness: bool | None
    completed: bool
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContrastPair:
    id: str
    prompt: str
    good: str
    bad: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ContrastPair":
        if not all(isinstance(d.get(k), str) and d[k].strip()
                   for k in ("id", "prompt", "good", "bad")):
            raise ValueError("Contrast pair requires nonempty id, prompt, good and bad")
        if d["good"] == d["bad"]:
            raise ValueError("A contrast pair's good and bad continuations must differ")
        return cls(**{key: d[key] for key in ("id", "prompt", "good", "bad")})


@dataclass
class Assessment:
    count: int
    labeled: int
    correct: int
    loop_count: int
    budget_exhausted: int
    request_errors: int
    incomplete: int
    findings: list[dict[str, Any]] = field(default_factory=list)
    anomaly_counts: dict[str, int] = field(default_factory=dict)
    by_tag: dict[str, dict[str, float | int | None]] = field(default_factory=dict)
    median_tokens_per_second: float | None = None
    p50_elapsed_ms: int | None = None
    p95_elapsed_ms: int | None = None

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.labeled if self.labeled else None

    @property
    def loop_rate(self) -> float:
        return self.loop_count / self.count if self.count else 0.0

    @property
    def cut_rate(self) -> float:
        return self.budget_exhausted / self.count if self.count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "accuracy": self.accuracy,
                "loop_rate": self.loop_rate, "cut_rate": self.cut_rate}
