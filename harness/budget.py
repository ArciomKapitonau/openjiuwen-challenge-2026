"""Token budget ledger.

Competition cost unit = output_tokens + 0.1 * input_tokens.
Reasoning tokens are counted as output even if the provider reports them
separately.  Provider cache is ignored (all input tokens are billed).
"""
from __future__ import annotations

import math
import os
import threading
import time


def cost_of(input_tokens: int, output_tokens: int, reasoning_tokens: int = 0, *,
            reasoning_in_output: bool = True) -> float:
    billed_output = output_tokens if reasoning_in_output else output_tokens + reasoning_tokens
    return (10 * billed_output + input_tokens) / 10


def estimate_tokens(text: str) -> int:
    """Cheap conservative token estimate (no tokenizer needed)."""
    if not text:
        return 0
    return max(64, len(text.encode("utf-8")) // 2 + 16)


class BudgetLedger:
    def __init__(self, total: float | None = None, reserve_frac: float = 0.02):
        if total is None:
            try:
                total = float(os.environ.get("BUDGET", "100000"))
            except ValueError:
                total = 100000.0
        if not math.isfinite(total) or total <= 0:
            raise ValueError("BUDGET must be a positive finite number")
        self.total = total
        self.reserve = min(max(300.0, total * reserve_frac), total * 0.30)
        mode = os.environ.get("HARNESS_REASONING_USAGE", "included")
        if mode not in {"included", "separate"}:
            raise ValueError("HARNESS_REASONING_USAGE must be included or separate")
        self.reasoning_in_output = mode == "included"
        self.spent = 0.0
        self.pending = 0.0  # reserved for in-flight calls
        self.calls = 0
        self.dead = False  # proxy stopped answering
        self._lock = threading.Lock()
        self.log: list[dict] = []

    @property
    def remaining(self) -> float:
        return self.total - self.spent - self.pending

    def usable(self) -> float:
        return self.remaining - self.reserve

    def try_reserve(self, amount: float) -> bool:
        with self._lock:
            if self.dead:
                return False
            if self.total - self.spent - self.pending - self.reserve < amount:
                return False
            self.pending += amount
            return True

    def commit(self, reserved: float, input_tokens: int, output_tokens: int,
               reasoning_tokens: int = 0, meta: dict | None = None) -> float:
        c = cost_of(input_tokens, output_tokens, reasoning_tokens,
                    reasoning_in_output=self.reasoning_in_output)
        with self._lock:
            self.pending = max(0.0, self.pending - reserved)
            self.spent += c
            self.calls += 1
            rec = {"t": round(time.time(), 2), "in": input_tokens, "out": output_tokens,
                   "reason": reasoning_tokens, "cost": round(c, 1),
                   "spent": round(self.spent, 1)}
            if meta:
                rec.update(meta)
            self.log.append(rec)
        return c

    def release(self, reserved: float):
        with self._lock:
            self.pending = max(0.0, self.pending - reserved)

    def mark_dead(self):
        with self._lock:
            self.dead = True
