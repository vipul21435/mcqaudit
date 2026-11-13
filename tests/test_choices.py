import random

import numpy as np
import pytest

from mcqaudit.choices import choices_only, featurise, make_folds, option_ngrams
from mcqaudit.data import Item

WORDS = [
    "apple",
    "river",
    "stone",
    "cloud",
    "lamp",
    "tiger",
    "violin",
    "copper",
    "meadow",
    "ferry",
    "glass",
    "orbit",
]


def _item(rng, i, plant, subject=None):
    opts = [" ".join(rng.choice(WORDS) for _ in range(3)) for _ in range(4)]
    answer = rng.randrange(4)
    if plant:
        opts[answer] = opts[answer] + " certainly"
    return Item(f"q{i}", "unused question", tuple(opts), answer, subject)


def test_ngrams():
    assert option_ngrams("Red apple pie") == ["red", "apple", "pie", "red apple", "apple pie"]
    assert option_ngrams("") == ["<empty>"]


def test_features_never_use_the_question():
    a = Item("a", "first question", ("x y", "z"), 0)
    b = Item("b", "a different question entirely", ("x y", "z"), 0)
    fa, fb = featurise([a], 64), featurise([b], 64)
    assert np.array_equal(fa.cols, fb.cols) and np.array_equal(fa.vals, fb.vals)
    assert fa.n_cols == 2 * 64


def test_planted_word_is_found():
    rng = random.Random(1)
    items = [_item(rng, i, plant=True, subject=f"s{i % 6}") for i in range(400)]
    res = choices_only(items, n_folds=5, n_features=2**10)
    assert res.folds == "grouped by subject"
    assert res.accuracy > 0.9
    assert res.ci_low > 0.25
    assert res.p_value < 1e-10


def test_false_positive_rate_without_a_cue():
    rng = random.Random(2)
    hits = 0
    trials = 20
    for t in range(trials):
        items = [_item(rng, f"{t}-{i}", plant=False) for i in range(120)]
        res = choices_only(items, n_folds=3, n_features=2**8, seed=t, max_iter=60)
        hits += res.p_value < 0.05
    # Binomial(20, 0.05) has mean 1; 5 or more has probability below 0.003.
    assert hits < 5


def test_folds_grouped_by_subject():
    items = [Item(f"q{i}", "q", ("a", "b"), 0, f"s{i % 7}") for i in range(70)]
    folds, how = make_folds(items, 5, 0, "subject")
    assert how == "grouped by subject"
    by_subject = {}
    for it, f in zip(items, folds, strict=True):
        by_subject.setdefault(it.subject, set()).add(f)
    assert all(len(v) == 1 for v in by_subject.values())
    folds, how = make_folds(items, 5, 0, "none")
    assert how.startswith("random")
    assert sorted(set(folds)) == [0, 1, 2, 3, 4]


def test_mixed_option_counts_are_masked():
    rng = random.Random(4)
    items = []
    for i in range(90):
        k = 3 if i % 2 else 4
        opts = tuple(rng.choice(WORDS) for _ in range(k))
        items.append(Item(f"q{i}", "q", opts, rng.randrange(k)))
    res = choices_only(items, n_folds=3, n_features=256)
    assert 0 <= res.accuracy <= 1
    assert res.chance == pytest.approx((45 / 3 + 45 / 4) / 90)


def test_rejects_too_few_items():
    with pytest.raises(ValueError):
        choices_only([Item("a", "q", ("x", "y"), 0)], n_folds=5)
