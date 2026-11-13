import pytest

from mcqaudit.text import char_ngrams, content_words, jaccard, normalise, parse_number, words


def test_normalise_and_words():
    assert normalise("  Hello\tWorld.  ") == "hello world"
    assert words("It's 3 o'clock!") == ["it's", "3", "o'clock"]
    assert content_words("The cat and the hat") == {"cat", "hat"}


@pytest.mark.parametrize(
    "text,value",
    [
        ("42", 42.0),
        ("-3.5", -3.5),
        ("1,200", 1200.0),
        ("3/8", 0.375),
        ("12 kg", 12.0),
        ("45%", 45.0),
        ("$7", 7.0),
        ("−2", -2.0),
        (".5", 0.5),
    ],
)
def test_parse_number(text, value):
    assert parse_number(text) == pytest.approx(value)


@pytest.mark.parametrize("text", ["four", "3 and 4", "between 2 and 5", "x = 3", "1/0", ""])
def test_parse_number_rejects(text):
    assert parse_number(text) is None


def test_ngrams_and_jaccard():
    assert char_ngrams("ab") == {" ab", "ab "}
    assert char_ngrams("") == {"  "}
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
