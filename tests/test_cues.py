import random

import pytest

from mcqaudit.cues import default_heuristics, evaluate, evaluate_all, longest, position, shortest
from mcqaudit.data import Item


def item(options, answer, qid="q"):
    return Item(qid, "question", tuple(options), answer)


def test_length_heuristics_and_ties():
    it = item(["aa", "bbbb", "c", "dddd"], 1)
    assert longest(it) == {1, 3}
    assert shortest(it) == {2}
    res = evaluate([it], "longest", longest)
    assert res.covered == 1
    assert res.credit == pytest.approx(0.5)


def test_abstains_when_every_option_ties():
    it = item(["ab", "cd", "ef"], 0)
    res = evaluate([it], "longest", longest)
    assert res.covered == 0
    assert res.p_value is None


def test_position_heuristic_abstains_on_short_items():
    pick_d = position(3)
    assert pick_d(item(["a", "b", "c"], 0)) == frozenset()
    assert pick_d(item(["a", "b", "c", "d"], 0)) == {3}
    assert pick_d.__name__ == "position_D"


def test_chance_and_interval_on_hand_built_items():
    items = [item(["x", "yy", "zzz", "w"], 2, f"q{i}") for i in range(8)]
    items += [item(["xxxx", "y", "z", "w"], 1, f"r{i}") for i in range(2)]
    res = evaluate(items, "longest", longest)
    assert res.covered == 10
    assert res.accuracy == pytest.approx(0.8)
    assert res.chance == pytest.approx(0.25)
    assert res.ci_low == pytest.approx(0.4902, abs=1e-4)
    assert res.z > 3
    assert res.p_value < 1e-3


def _random_item(rng, qid, plant):
    n = 4
    lengths = [rng.randint(3, 30) for _ in range(n)]
    answer = rng.randrange(n)
    if plant and rng.random() < 0.6:
        lengths[answer] = max(lengths) + rng.randint(1, 10)
    return Item(qid, "q", tuple("w" * L for L in lengths), answer)


def test_planted_length_cue_is_found():
    rng = random.Random(5)
    items = [_random_item(rng, f"q{i}", plant=True) for i in range(300)]
    best = evaluate_all(items)[0]
    assert best.name == "longest option"
    assert best.ci_low > 0.25
    assert best.p_value < 1e-6


def test_false_positive_rate_without_a_cue():
    rng = random.Random(99)
    trials, hits = 300, 0
    for t in range(trials):
        items = [_random_item(rng, f"q{t}-{i}", plant=False) for i in range(150)]
        res = evaluate(items, "longest", longest)
        hits += res.p_value < 0.05
    # Binomial(300, 0.05): mean 15, sd about 3.8.
    assert 4 <= hits <= 27


def test_default_set_and_ordering():
    names = list(default_heuristics(4))
    assert names[:4] == ["always A", "always B", "always C", "always D"]
    items = [item(["a", "bb", "c"], 0, f"q{i}") for i in range(5)]
    results = evaluate_all(items)
    assert results[0].name == "always A"
    assert results[0].accuracy == pytest.approx(1.0)
    assert "always D" not in {r.name for r in results}
    lifts = [r.lift for r in results]
    assert lifts == sorted(lifts, reverse=True)
    never = evaluate_all(items, {"always D": position(3), "always A": position(0)})
    assert never[-1].name == "always D" and never[-1].covered == 0
