"""Recommendation dataclass and common types."""

from dataclasses import dataclass, field


@dataclass
class Recommendation:
    slug: str
    title: str
    year: int | None
    score: float
    reasons: list[str]
    warnings: list[str] = field(default_factory=list)
