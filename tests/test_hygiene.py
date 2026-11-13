from mcqaudit.data import Item
from mcqaudit.hygiene import hygiene, shingles, split_overlap


def it(qid, question, options, answer):
    return Item(qid, question, tuple(options), answer)


def test_hygiene_counts_planted_defects():
    items = [
        it("ok", "What colour is the sky?", ["blue", "green", "red", "pink"], 0),
        it("dup-opt", "Pick one", ["yes", "Yes.", "no", "maybe"], 2),
        it("empty", "Pick another", ["a1", "", "c1", "d1"], 0),
        it(
            "leak",
            "The capital of France, Paris, is on which river?",
            ["Paris", "Seine", "x", "y"],
            0,
        ),
        it(
            "leak-only",
            "Is copper a metal or a gas? copper",
            ["copper", "argon", "neon", "xenon"],
            0,
        ),
        it("dq1", "Same question here", ["a", "b", "c", "d"], 0),
        it("dq2", "same  question here.", ["a", "b", "c", "d"], 1),
        it("three", "Short item", ["p", "q", "r"], 1),
    ]
    rep = hygiene(items)
    assert rep.duplicate_options == 1
    assert rep.empty_options == 1
    assert rep.answer_in_question == 2
    assert rep.answer_only_in_question == 2
    assert rep.duplicate_question_groups == 1
    assert rep.duplicate_question_items == 2
    assert rep.conflicting_duplicates == 1
    assert rep.option_counts == {3: 1, 4: 7}
    assert rep.off_mode_option_count == 1
    assert rep.examples["duplicate_options"] == ["dup-opt"]


def test_short_answers_do_not_count_as_leaks():
    rep = hygiene([it("x", "Is a an article?", ["a", "an", "the", "of"], 0)])
    assert rep.answer_in_question == 0


def test_clean_items():
    rep = hygiene([it("x", "q one", ["a", "b"], 0), it("y", "q two", ["c", "d"], 1)])
    assert rep.duplicate_options == rep.empty_options == rep.duplicate_question_groups == 0
    assert hygiene([]).option_counts == {}


def test_shingles():
    assert shingles("one two three four") == {"one two three", "two three four"}
    assert shingles("hi there") == {"hi", "there"}


def test_split_overlap_exact_and_near():
    a = [
        it("a1", "Which planet is known as the red planet in our solar system?", ["x", "y"], 0),
        it("a2", "What is the boiling point of water at sea level in celsius?", ["x", "y"], 0),
        it("a3", "Name the largest ocean on the planet Earth today", ["x", "y"], 0),
    ]
    b = [
        it("b1", "Which planet is known as the red planet in our solar system?", ["x", "y"], 0),
        it(
            "b2",
            "What is the boiling point of water at sea level in degrees celsius?",
            ["x", "y"],
            0,
        ),
        it("b3", "Who wrote the play about a prince of Denmark?", ["x", "y"], 0),
    ]
    rep = split_overlap(a, b, threshold=0.7)
    assert rep.exact == 1
    assert rep.near == 2
    assert rep.examples[0][:2] == ("a1", "b1")
    strict = split_overlap(a, b, threshold=0.99)
    assert strict.near == 1
