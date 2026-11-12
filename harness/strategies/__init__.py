"""Strategy registry.  Add a module here and register it in STRATEGIES."""
from __future__ import annotations

from ..strategy import check_strategy
from .baseline import Baseline

STRATEGIES = {
    "baseline": Baseline,
}


def get(name: str):
    if name not in STRATEGIES:
        raise SystemExit(f"unknown strategy {name!r}; known: {', '.join(sorted(STRATEGIES))}")
    s = STRATEGIES[name]()
    problems = check_strategy(s)
    if problems:
        raise SystemExit(f"strategy {name!r} violates the contract: {problems}")
    return s
