import random
from collections import Counter

from mcqaudit.data import Item, load
from mcqaudit.keyaudit import audit_key
from mcqaudit.shuffle import balance_positions, shuffle_options, write_jsonl


def _items(n=200, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        k = rng.choice([3, 4, 5])
        opts = tuple(f"opt-{i}-{j}" for j in range(k))
        out.append(Item(f"q{i}", f"question {i}", opts, 0 if rng.random() < 0.7 else k - 1, "s"))
    return out


def test_shuffle_preserves_answer_text_and_options():
    items = _items()
    shuffled = shuffle_options(items, seed=3)
    for a, b in zip(items, shuffled, strict=True):
        assert b.answer_text == a.answer_text
        assert sorted(b.options) == sorted(a.options)
        assert (b.id, b.question, b.subject) == (a.id, a.question, a.subject)
    assert shuffle_options(items, seed=3) == shuffled
    assert shuffle_options(items, seed=4) != shuffled


def test_shuffle_removes_a_planted_position_bias():
    items = _items(600)
    assert audit_key(items).positions.p_value < 1e-10
    assert audit_key(shuffle_options(items, 1)).positions.p_value > 1e-3


def test_balance_makes_counts_even():
    items = _items(301)
    balanced = balance_positions(items, seed=0)
    for a, b in zip(items, balanced, strict=True):
        assert b.answer_text == a.answer_text
        rest_a = [o for j, o in enumerate(a.options) if j != a.answer]
        rest_b = [o for j, o in enumerate(b.options) if j != b.answer]
        assert rest_a == rest_b
    for k in (3, 4, 5):
        counts = Counter(b.answer for b in balanced if b.n_options == k)
        assert max(counts.values()) - min(counts.values()) <= 1
        assert set(counts) == set(range(k))


def test_write_jsonl_roundtrip(tmp_path):
    items = _items(20)
    path = tmp_path / "out.jsonl"
    write_jsonl(items, path)
    back = load(path, answer_base=0).items
    assert [(b.id, b.options, b.answer, b.subject) for b in back] == [
        (a.id, a.options, a.answer, a.subject) for a in items
    ]
