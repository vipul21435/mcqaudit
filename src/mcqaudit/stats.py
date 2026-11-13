"""Small statistical routines, written on the standard library.

The chi-square tail is the regularized upper incomplete gamma function,
computed by its power series below ``a + 1`` and by a continued fraction above
it (the split used in most numerical libraries). The tests compare every
routine here with scipy, which is a test-only dependency.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist

_EPS = 1e-15
_TINY = 1e-300
_MAX_ITER = 100_000


def _log_prefactor(a: float, x: float) -> float:
    return -x + a * math.log(x) - math.lgamma(a)


def _gamma_p_series(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) by its power series."""
    term = 1.0 / a
    total = term
    denom = a
    for _ in range(_MAX_ITER):
        denom += 1.0
        term *= x / denom
        total += term
        if abs(term) < abs(total) * _EPS:
            break
    return total * math.exp(_log_prefactor(a, x))


def _gamma_q_fraction(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) by a continued fraction (Lentz)."""
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for i in range(1, _MAX_ITER):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _TINY:
            d = _TINY
        c = b + an / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(_log_prefactor(a, x)) * h


def gamma_q(a: float, x: float) -> float:
    """Regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x)."""
    if a <= 0:
        raise ValueError("a must be positive")
    if x < 0:
        raise ValueError("x must be non-negative")
    if x == 0:
        return 1.0
    if x < a + 1.0:
        return max(0.0, 1.0 - _gamma_p_series(a, x))
    return _gamma_q_fraction(a, x)


def chi2_sf(x: float, df: float) -> float:
    """Upper tail P(X >= x) of a chi-square distribution with ``df`` degrees of freedom."""
    if df <= 0:
        raise ValueError("df must be positive")
    if x <= 0:
        return 1.0
    return gamma_q(df / 2.0, x / 2.0)


def normal_two_sided_p(z: float) -> float:
    """Two-sided p-value of a standard normal statistic."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def normal_upper_p(z: float) -> float:
    """One-sided p-value P(Z >= z) of a standard normal statistic."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def z_for_confidence(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


def wilson_interval(successes: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    ``successes`` may be fractional (expected credit under random tie-breaking);
    the interval is then slightly conservative.
    """
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= successes <= n:
        raise ValueError("successes must lie between 0 and n")
    z = z_for_confidence(confidence)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z / denom * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def chi_square_statistic(observed: Sequence[float], expected: Sequence[float]) -> float:
    """Pearson's goodness-of-fit statistic, sum (O - E)^2 / E."""
    if len(observed) != len(expected):
        raise ValueError("observed and expected differ in length")
    stat = 0.0
    for o, e in zip(observed, expected, strict=True):
        if e <= 0:
            raise ValueError("expected counts must be positive")
        stat += (o - e) ** 2 / e
    return stat


def multinomial_outcomes(n: int, k: int) -> int:
    """Number of ways to put ``n`` items into ``k`` cells, C(n + k - 1, k - 1)."""
    return math.comb(n + k - 1, k - 1)


def multinomial_exact_p(
    counts: Sequence[int],
    probs: Sequence[float] | None = None,
    *,
    max_outcomes: int = 500_000,
) -> float:
    """Exact multinomial goodness-of-fit p-value.

    The p-value is the total probability, under the null cell probabilities, of
    every table with the same total whose probability is at most that of the
    observed table. With two cells this is the two-sided exact binomial test.
    Raises ``ValueError`` when the number of tables exceeds ``max_outcomes``.
    """
    k = len(counts)
    if k < 2:
        raise ValueError("need at least two cells")
    if any(c < 0 for c in counts):
        raise ValueError("counts must be non-negative")
    n = int(sum(counts))
    if probs is None:
        probs = [1.0 / k] * k
    if len(probs) != k or any(p <= 0 for p in probs):
        raise ValueError("probs must be positive, one per cell")
    total_p = float(sum(probs))
    probs = [p / total_p for p in probs]
    if n == 0:
        return 1.0
    outcomes = multinomial_outcomes(n, k)
    if outcomes > max_outcomes:
        raise ValueError(f"{outcomes} tables exceed max_outcomes={max_outcomes}")

    log_n_fact = math.lgamma(n + 1)
    log_fact = [math.lgamma(j + 1) for j in range(n + 1)]
    terms = [[j * math.log(p) - log_fact[j] for j in range(n + 1)] for p in probs]
    observed = sum(terms[i][c] for i, c in enumerate(counts))
    # Relative tolerance on the probability, as in common exact-test implementations.
    threshold = observed + 1e-7

    total = 0.0
    last = terms[k - 1]
    second = terms[k - 2]

    def walk(cell: int, remaining: int, acc: float) -> None:
        nonlocal total
        if cell == k - 2:
            for j in range(remaining + 1):
                lp = acc + second[j] + last[remaining - j]
                if lp <= threshold:
                    total += math.exp(lp + log_n_fact)
            return
        row = terms[cell]
        for j in range(remaining + 1):
            walk(cell + 1, remaining - j, acc + row[j])

    walk(0, n, 0.0)
    return min(1.0, total)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def credit_upper_p(
    items: Sequence[tuple[int, int]], observed: float, *, max_units: int = 2_000_000
) -> float | None:
    """Exact P(total credit >= observed) for independent items under the no-cue null.

    Each pair ``(k, c)`` is an item with ``k`` options on which a heuristic names
    ``c`` options: it earns ``1/c`` with probability ``c/k`` and 0 otherwise. The
    credits are put on a common grid of ``1/L`` (``L`` the least common multiple
    of the ``c``) and the distributions are convolved. Returns ``None`` when the
    grid would exceed ``max_units`` cells times items.
    """
    if not items:
        return 1.0
    lcm = 1
    for _, c in items:
        lcm = lcm * c // math.gcd(lcm, c)
    values = [lcm // c for _, c in items]
    total = sum(values)
    if total * len(items) > max_units:
        return None
    dist = [0.0] * (total + 1)
    dist[0] = 1.0
    top = 0
    for (k, c), v in zip(items, values, strict=True):
        q = c / k
        for s in range(top, -1, -1):
            mass = dist[s]
            if mass:
                dist[s + v] += mass * q
                dist[s] = mass * (1.0 - q)
        top += v
    threshold = round(observed * lcm - 1e-9)
    return min(1.0, sum(dist[threshold:]))
