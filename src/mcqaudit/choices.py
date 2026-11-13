"""Choices-only classifier: how much the options alone reveal about the key.

Each item becomes one row: for every position ``j`` the word unigrams and
bigrams of option ``j`` are hashed (signed feature hashing, CRC-32) into their
own block of ``n_features`` columns. The question text is never read. A
multinomial logistic regression over positions is fitted on the training folds
and scored on the held-out fold; positions an item does not have are masked.

Folds are grouped by subject when the data has at least as many subjects as
folds, so that the model is always tested on subjects it has not seen; without
subjects the folds are a seeded random split. Training never sees an item whose
normalised option set also occurs in the held-out fold, so repeated items are
not answered from memory. Its out-of-fold accuracy above
chance is a lower bound on what the options give away; the report also gives
the accuracy of always picking the training folds' most common position, so the
part due to key imbalance alone can be read off.
"""

from __future__ import annotations

import math
import random
import warnings
import zlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .data import Item
from .logreg import SoftmaxRegression, SparseRows
from .stats import normal_upper_p, wilson_interval
from .text import normalise, words


def option_ngrams(text: str, max_n: int = 2) -> list[str]:
    toks = words(text)
    grams = list(toks)
    for n in range(2, max_n + 1):
        grams += [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        grams = ["<empty>"]
    return grams


def featurise(items: Sequence[Item], n_features: int = 2**12, max_n: int = 2) -> SparseRows:
    """Hashed n-gram features of the options, one block per position, as a sparse matrix."""
    width = max((it.n_options for it in items), default=2)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    cache: dict[str, int] = {}
    for i, item in enumerate(items):
        for j, option in enumerate(item.options):
            counts: dict[int, float] = defaultdict(float)
            for gram in sorted(set(option_ngrams(option, max_n))):
                h = cache.get(gram)
                if h is None:
                    h = zlib.crc32(gram.encode("utf-8"))
                    cache[gram] = h
                sign = 1.0 if (h >> 31) & 1 else -1.0
                counts[j * n_features + (h % n_features)] += sign
            for col, val in counts.items():
                if val:
                    rows.append(i)
                    cols.append(col)
                    vals.append(val)
    return SparseRows(
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(vals, dtype=float),
        len(items),
        width * n_features,
    )


def make_folds(items: Sequence[Item], n_folds: int, seed: int, group: str) -> tuple[list[int], str]:
    """Fold number per item, and how the folds were built."""
    n = len(items)
    subjects = Counter(it.subject for it in items if it.subject)
    if group == "subject" and len(subjects) >= n_folds and all(it.subject for it in items):
        # Largest subjects first, each to the currently smallest fold.
        rng = random.Random(seed)
        names = sorted(subjects, key=lambda s: (-subjects[s], rng.random()))
        sizes = [0] * n_folds
        fold_of: dict[str, int] = {}
        for name in names:
            f = min(range(n_folds), key=lambda j: (sizes[j], j))
            fold_of[name] = f
            sizes[f] += subjects[name]
        return [fold_of[it.subject] for it in items], "grouped by subject"  # type: ignore[index]
    order = list(range(n))
    random.Random(seed).shuffle(order)
    folds = [0] * n
    for rank, i in enumerate(order):
        folds[i] = rank % n_folds
    return folds, "random (no subject grouping)"


@dataclass
class ChoicesOnlyResult:
    n_items: int
    n_folds: int
    folds: str
    correct: int
    accuracy: float
    ci_low: float
    ci_high: float
    chance: float
    z: float
    p_value: float
    majority_accuracy: float
    n_features: int
    C: float
    test_items_seen_in_training: int = 0
    max_iter: int = 0
    folds_at_iteration_cap: int = 0

    @property
    def lift(self) -> float:
        return self.accuracy - self.chance


def choices_only(
    items: Sequence[Item],
    *,
    n_folds: int = 5,
    seed: int = 0,
    C: float = 1.0,
    n_features: int = 2**12,
    max_n: int = 2,
    group: str = "subject",
    confidence: float = 0.95,
    max_iter: int = 300,
    ftol: float = 1e-6,
) -> ChoicesOnlyResult:
    """Cross-validated accuracy of a classifier that sees only the options."""
    n = len(items)
    if n < n_folds or n_folds < 2:
        raise ValueError("need at least two folds and one item per fold")
    width = max(it.n_options for it in items)
    x = featurise(items, n_features, max_n)
    y = np.array([it.answer for it in items])
    mask = np.zeros((n, width), dtype=bool)
    for i, it in enumerate(items):
        mask[i, : it.n_options] = True
    folds, how = make_folds(items, n_folds, seed, group)
    fold_arr = np.array(folds)

    pred = np.full(n, -1)
    majority_hits = 0
    at_cap = 0
    keys = [tuple(sorted(normalise(o) for o in it.options)) for it in items]
    seen = 0
    for f in range(n_folds):
        test = np.flatnonzero(fold_arr == f)
        test_keys = {keys[i] for i in test}
        # An item whose option set also occurs in the test fold is left out of training,
        # so repeated items cannot be answered from memory.
        train_all = np.flatnonzero(fold_arr != f)
        train_keys = {keys[i] for i in train_all}
        seen += sum(keys[i] in train_keys for i in test)
        train = np.array([i for i in train_all if keys[i] not in test_keys], dtype=int)
        if len(test) == 0 or len(train) == 0:
            continue
        model = SoftmaxRegression(width, C=C, max_iter=max_iter, ftol=ftol)
        model.fit(x.take(train), y[train], mask[train])
        at_cap += not model.converged_
        pred[test] = model.predict(x.take(test), mask[test])
        top = Counter(y[train].tolist()).most_common(1)[0][0]
        majority_hits += int(np.sum(y[test] == top))

    if at_cap:
        warnings.warn(
            f"L-BFGS stopped at max_iter={max_iter} in {at_cap} of {n_folds} folds "
            "before converging; raise max_iter for the optimum",
            RuntimeWarning,
            stacklevel=2,
        )
    correct = int(np.sum(pred == y))
    ks = np.array([it.n_options for it in items], dtype=float)
    chance_sum = float(np.sum(1.0 / ks))
    var_sum = float(np.sum(1.0 / ks * (1.0 - 1.0 / ks)))
    z = (correct - chance_sum) / math.sqrt(var_sum)
    lo, hi = wilson_interval(correct, n, confidence)
    return ChoicesOnlyResult(
        n_items=n,
        n_folds=n_folds,
        folds=how,
        correct=correct,
        accuracy=correct / n,
        ci_low=lo,
        ci_high=hi,
        chance=chance_sum / n,
        z=z,
        p_value=normal_upper_p(z),
        majority_accuracy=majority_hits / n,
        n_features=n_features,
        C=C,
        test_items_seen_in_training=seen,
        max_iter=max_iter,
        folds_at_iteration_cap=at_cap,
    )
