"""Audit of the answer key: where the correct option sits and in what order.

Position balance is tested separately for each option count (an item with three
options cannot have its answer at D), with Pearson's chi-square test or, when
the number of possible tables is small enough to enumerate, the exact
multinomial test. Strata are combined by summing their chi-square statistics
and degrees of freedom.

Order is tested with the k-category runs test: a run is a maximal block of
consecutive items with the same answer position. Conditional on the counts
n_j (sum n), under random order

    E[R]   = (n(n + 1) - S2) / n
    Var[R] = (S2 (S2 + n(n + 1)) - 2 n S3 - n^3) / (n^2 (n - 1))

with S2 = sum n_j^2 and S3 = sum n_j^3; with two categories this is the
Wald-Wolfowitz runs test. Too few runs means answers cluster; too many means
the key avoids repeating a position, as hand-made keys often do.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from .data import Item, position_label
from .stats import (
    chi2_sf,
    chi_square_statistic,
    holm_adjust,
    multinomial_exact_p,
    multinomial_outcomes,
    normal_two_sided_p,
)

DEFAULT_EXACT_LIMIT = 200_000


@dataclass
class PositionTest:
    """Goodness of fit of the answer positions to a uniform key, for one option count."""

    n_options: int
    n: int
    counts: list[int]
    expected: list[float]
    statistic: float
    df: int
    p_value: float
    method: str

    @property
    def shares(self) -> list[float]:
        return [c / self.n if self.n else 0.0 for c in self.counts]

    @property
    def top_position(self) -> str:
        best = max(range(self.n_options), key=lambda j: self.counts[j])
        return position_label(best)


@dataclass
class PositionSummary:
    """Position tests per option count, and their combination."""

    n: int
    strata: list[PositionTest]
    statistic: float
    df: int
    p_value: float
    method: str


@dataclass
class RunsTest:
    """Runs test on the sequence of answer positions, in file order."""

    n: int
    runs: int
    expected_runs: float
    sd_runs: float
    z: float | None
    p_value: float | None

    @property
    def repeats(self) -> int:
        """Consecutive pairs with the same answer position."""
        return max(0, self.n - self.runs)


@dataclass
class SubjectKey:
    subject: str
    n: int
    positions: PositionSummary
    runs: RunsTest
    p_holm: float = 1.0


@dataclass
class KeyAudit:
    n_items: int
    positions: PositionSummary
    runs: RunsTest
    subjects: list[SubjectKey] = field(default_factory=list)


def position_test(
    answers: Sequence[int],
    n_options: int,
    *,
    method: str = "auto",
    exact_limit: int = DEFAULT_EXACT_LIMIT,
) -> PositionTest:
    """Test whether 0-based ``answers`` are uniform over ``n_options`` positions.

    ``method`` is ``"chi-square"``, ``"exact"`` or ``"auto"`` (exact when at most
    ``exact_limit`` tables are possible).
    """
    if n_options < 2:
        raise ValueError("n_options must be at least 2")
    counts = [0] * n_options
    for a in answers:
        if not 0 <= a < n_options:
            raise ValueError(f"answer {a} out of range for {n_options} options")
        counts[a] += 1
    n = len(answers)
    if n == 0:
        return PositionTest(n_options, 0, counts, [0.0] * n_options, 0.0, n_options - 1, 1.0, "")
    expected = [n / n_options] * n_options
    stat = chi_square_statistic(counts, expected)
    df = n_options - 1

    if method == "auto":
        method = "exact" if multinomial_outcomes(n, n_options) <= exact_limit else "chi-square"
    if method == "exact":
        p = multinomial_exact_p(counts, max_outcomes=max(exact_limit, 1))
        label = "exact multinomial"
    elif method == "chi-square":
        p = chi2_sf(stat, df)
        label = "chi-square"
    else:
        raise ValueError("method must be 'auto', 'exact' or 'chi-square'")
    return PositionTest(n_options, n, counts, expected, stat, df, p, label)


def position_summary(
    items: Sequence[Item], *, method: str = "auto", exact_limit: int = DEFAULT_EXACT_LIMIT
) -> PositionSummary:
    """Position tests for each option count present in ``items``, and their sum."""
    groups: dict[int, list[int]] = defaultdict(list)
    for item in items:
        groups[item.n_options].append(item.answer)
    strata = [
        position_test(groups[k], k, method=method, exact_limit=exact_limit) for k in sorted(groups)
    ]
    if not strata:
        return PositionSummary(0, [], 0.0, 0, 1.0, "")
    if len(strata) == 1:
        s = strata[0]
        return PositionSummary(s.n, strata, s.statistic, s.df, s.p_value, s.method)
    stat = sum(s.statistic for s in strata)
    df = sum(s.df for s in strata)
    return PositionSummary(
        len(items), strata, stat, df, chi2_sf(stat, df), "chi-square summed over option counts"
    )


def runs_test(sequence: Sequence[Hashable]) -> RunsTest:
    """k-category runs test with the normal approximation, two-sided."""
    n = len(sequence)
    if n == 0:
        return RunsTest(0, 0, 0.0, 0.0, None, None)
    runs = 1 + sum(1 for a, b in pairwise(sequence) if a != b)
    counts: dict[Hashable, int] = defaultdict(int)
    for s in sequence:
        counts[s] += 1
    s2 = sum(c * c for c in counts.values())
    s3 = sum(c**3 for c in counts.values())
    expected = (n * (n + 1) - s2) / n
    if n < 2:
        return RunsTest(n, runs, expected, 0.0, None, None)
    var = (s2 * (s2 + n * (n + 1)) - 2 * n * s3 - n**3) / (n * n * (n - 1))
    if var <= 0:
        return RunsTest(n, runs, expected, 0.0, None, None)
    sd = math.sqrt(var)
    z = (runs - expected) / sd
    return RunsTest(n, runs, expected, sd, z, normal_two_sided_p(z))


def audit_key(
    items: Sequence[Item],
    *,
    by_subject: bool = True,
    method: str = "auto",
    exact_limit: int = DEFAULT_EXACT_LIMIT,
) -> KeyAudit:
    """Position balance and order of the answer key, overall and by subject."""
    positions = position_summary(items, method=method, exact_limit=exact_limit)
    runs = runs_test([item.answer for item in items])
    subjects: list[SubjectKey] = []
    if by_subject and any(item.subject for item in items):
        grouped: dict[str, list[Item]] = defaultdict(list)
        for item in items:
            grouped[item.subject or "(none)"].append(item)
        for name in sorted(grouped):
            group = grouped[name]
            subjects.append(
                SubjectKey(
                    name,
                    len(group),
                    position_summary(group, method=method, exact_limit=exact_limit),
                    runs_test([item.answer for item in group]),
                )
            )
        adjusted = holm_adjust([s.positions.p_value for s in subjects])
        for subject, p in zip(subjects, adjusted, strict=True):
            subject.p_holm = p
    return KeyAudit(len(items), positions, runs, subjects)
