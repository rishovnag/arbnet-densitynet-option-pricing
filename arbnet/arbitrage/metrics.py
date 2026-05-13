"""Aggregated arbitrage metrics across many snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Iterable

from .checks import StaticArbReport


@dataclass
class AggregateArbReport:
    """Aggregate of static arbitrage reports across snapshots."""

    n_snapshots: int
    butterfly_total_violations: int
    calendar_total_violations: int
    tail_total_violations: int
    butterfly_total_tests: int
    calendar_total_tests: int
    tail_total_tests: int

    @property
    def butterfly_rate(self) -> float:
        return self.butterfly_total_violations / max(1, self.butterfly_total_tests)

    @property
    def calendar_rate(self) -> float:
        return self.calendar_total_violations / max(1, self.calendar_total_tests)

    @property
    def tail_rate(self) -> float:
        return self.tail_total_violations / max(1, self.tail_total_tests)

    @property
    def total_rate(self) -> float:
        n = self.butterfly_total_violations + self.calendar_total_violations + self.tail_total_violations
        d = self.butterfly_total_tests + self.calendar_total_tests + self.tail_total_tests
        return n / max(1, d)


def arbitrage_violation_rate(reports: Iterable[StaticArbReport]) -> AggregateArbReport:
    """Aggregate per-snapshot reports into an across-snapshot summary."""
    reports = list(reports)
    n = len(reports)
    # We re-derive raw counts from the rates because StaticArbReport keeps both.
    # Each report has count and rate; total = count / rate. To stay safe, use counts.
    bf_v = sum(r.butterfly_count for r in reports)
    cal_v = sum(r.calendar_count for r in reports)
    tail_v = sum(r.tail_count for r in reports)
    bf_t = sum(int(round(r.butterfly_count / r.butterfly_rate)) if r.butterfly_rate > 0 else 0 for r in reports)
    cal_t = sum(int(round(r.calendar_count / r.calendar_rate)) if r.calendar_rate > 0 else 0 for r in reports)
    tail_t = sum(int(round(r.tail_count / r.tail_rate)) if r.tail_rate > 0 else 0 for r in reports)
    # If a report had zero violations, we can't recover its total this way; this is
    # a known limitation. For full reproducibility we recommend passing totals directly.
    return AggregateArbReport(
        n_snapshots=n,
        butterfly_total_violations=bf_v,
        calendar_total_violations=cal_v,
        tail_total_violations=tail_v,
        butterfly_total_tests=bf_t,
        calendar_total_tests=cal_t,
        tail_total_tests=tail_t,
    )
