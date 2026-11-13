"""Question-free heuristics and how far they score above chance.

A heuristic looks at one item and names the options it would pick. When it
names several (a tie), it is scored by its expected credit under random
tie-breaking, ``1/|C|`` when the answer is in the set ``C``; when it names every
option, or none, it abstains on that item. Accuracy is reported over the items
it covers, with a Wilson interval.

Against chance: if the key position were independent of what the heuristic
looks at, an item with ``k`` options would give credit with mean ``1/k`` and
variance ``1/(k|C|) - 1/k^2``. The z statistic sums both over covered items.
The one-sided p-value is exact (the per-item credit distributions convolved)
when at most ``EXACT_MAX_ITEMS`` items are covered, and uses the normal
approximation above that.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .data import LETTERS, Item
from .stats import credit_upper_p, normal_upper_p, wilson_interval
from .text import char_ngrams, content_words, jaccard, normalise, parse_number

Heuristic = Callable[[Item], frozenset[int]]

EXACT_MAX_ITEMS = 500


def _extremes(values: Sequence[float], pick_max: bool) -> frozenset[int]:
    target = max(values) if pick_max else min(values)
    return frozenset(i for i, v in enumerate(values) if v == target)


def _length(option: str) -> int:
    return len(option.strip())


def longest(item: Item) -> frozenset[int]:
    """The option with the most characters."""
    return _extremes([_length(o) for o in item.options], pick_max=True)


def shortest(item: Item) -> frozenset[int]:
    """The option with the fewest characters."""
    return _extremes([_length(o) for o in item.options], pick_max=False)


def position(k: int) -> Heuristic:
    """Always the option at 0-based position ``k``; abstains when an item has fewer options."""

    def pick(item: Item) -> frozenset[int]:
        return frozenset({k}) if k < item.n_options else frozenset()

    pick.__name__ = f"position_{LETTERS[k]}"
    pick.__doc__ = f"Always option {LETTERS[k]}."
    return pick


def _overlaps(item: Item) -> list[int]:
    question = content_words(item.question)
    return [len(content_words(o) & question) for o in item.options]


def most_overlap(item: Item) -> frozenset[int]:
    """The option sharing the most content words with the question."""
    return _extremes(_overlaps(item), pick_max=True)


def least_overlap(item: Item) -> frozenset[int]:
    """The option sharing the fewest content words with the question."""
    return _extremes(_overlaps(item), pick_max=False)


_ALL_ABOVE = re.compile(
    r"\ball (of )?(the )?(above|options|choices|answers|of these)\b|"
    r"\ball (are|of them are) (correct|true)\b"
)
_NONE_ABOVE = re.compile(
    r"\bnone (of )?(the )?(above|options|choices|answers|of these)\b|"
    r"\bnone (are|is|of them are) (correct|true)\b"
)
_BOTH = re.compile(r"\bboth\b")


def _matching(pattern: re.Pattern[str]) -> Heuristic:
    def pick(item: Item) -> frozenset[int]:
        return frozenset(i for i, o in enumerate(item.options) if pattern.search(normalise(o)))

    return pick


all_of_the_above = _matching(_ALL_ABOVE)
all_of_the_above.__doc__ = 'An option of the "all of the above" kind.'
none_of_the_above = _matching(_NONE_ABOVE)
none_of_the_above.__doc__ = 'An option of the "none of the above" kind.'
both_option = _matching(_BOTH)
both_option.__doc__ = 'An option containing "both" ("both A and B").'


def middle_number(item: Item) -> frozenset[int]:
    """When every option is a single number: the option(s) holding the median value."""
    values = [parse_number(o) for o in item.options]
    if any(v is None for v in values) or len(set(values)) < 3:
        return frozenset()
    distinct = sorted(set(values))  # type: ignore[type-var]
    m = len(distinct)
    middle = {distinct[(m - 1) // 2], distinct[m // 2]}
    return frozenset(i for i, v in enumerate(values) if v in middle)


def _similarity_to_others(item: Item) -> list[float]:
    grams = [char_ngrams(o) for o in item.options]
    k = len(grams)
    return [
        sum(jaccard(grams[i], grams[j]) for j in range(k) if j != i) / (k - 1) for i in range(k)
    ]


def most_similar(item: Item) -> frozenset[int]:
    """The option most similar to the other options (mean character-trigram Jaccard)."""
    return _extremes(_similarity_to_others(item), pick_max=True)


def odd_one_out(item: Item) -> frozenset[int]:
    """The option least similar to the other options."""
    return _extremes(_similarity_to_others(item), pick_max=False)


def default_heuristics(max_options: int) -> dict[str, Heuristic]:
    """The fixed set of heuristics, named as they appear in reports."""
    table: dict[str, Heuristic] = {
        f"always {LETTERS[k]}": position(k) for k in range(max(max_options, 2))
    }
    table["longest option"] = longest
    table["shortest option"] = shortest
    table["most question overlap"] = most_overlap
    table["least question overlap"] = least_overlap
    table["all of the above"] = all_of_the_above
    table["none of the above"] = none_of_the_above
    table["both ..."] = both_option
    table["middle number"] = middle_number
    table["most similar to others"] = most_similar
    table["odd one out"] = odd_one_out
    return table


@dataclass
class CueResult:
    name: str
    n_items: int
    covered: int
    credit: float
    accuracy: float
    ci_low: float
    ci_high: float
    chance: float
    z: float | None
    p_value: float | None
    p_exact: bool = False

    @property
    def lift(self) -> float:
        """Accuracy minus chance, in proportion units."""
        return self.accuracy - self.chance


def evaluate(
    items: Sequence[Item], name: str, heuristic: Heuristic, *, confidence: float = 0.95
) -> CueResult:
    """Score one heuristic on ``items``."""
    covered = 0
    credit = 0.0
    chance_sum = 0.0
    var_sum = 0.0
    pairs: list[tuple[int, int]] = []
    for item in items:
        picks = heuristic(item)
        k = item.n_options
        if not picks or len(picks) >= k:
            continue
        covered += 1
        size = len(picks)
        pairs.append((k, size))
        if item.answer in picks:
            credit += 1.0 / size
        chance_sum += 1.0 / k
        var_sum += 1.0 / (k * size) - 1.0 / (k * k)
    if covered == 0:
        return CueResult(name, len(items), 0, 0.0, 0.0, 0.0, 1.0, 0.0, None, None)
    accuracy = credit / covered
    lo, hi = wilson_interval(credit, covered, confidence)
    z = p = None
    exact = False
    if var_sum > 0:
        z = (credit - chance_sum) / math.sqrt(var_sum)
        if covered <= EXACT_MAX_ITEMS:
            p = credit_upper_p(pairs, credit)
            exact = p is not None
        if p is None:
            p = normal_upper_p(z)
    return CueResult(
        name, len(items), covered, credit, accuracy, lo, hi, chance_sum / covered, z, p, exact
    )


def evaluate_all(
    items: Sequence[Item],
    heuristics: dict[str, Heuristic] | None = None,
    *,
    confidence: float = 0.95,
) -> list[CueResult]:
    """Score every heuristic, best lift first."""
    if heuristics is None:
        heuristics = default_heuristics(max((i.n_options for i in items), default=4))
    results = [evaluate(items, n, h, confidence=confidence) for n, h in heuristics.items()]
    results.sort(key=lambda r: (r.covered == 0, -r.lift))
    return results
