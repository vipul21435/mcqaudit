"""Text normalisation shared by the heuristics, hygiene checks and features."""

from __future__ import annotations

import re
import unicodedata


def _mark_ranges() -> str:
    """Character-class ranges of the combining marks (category M) in the BMP."""
    ranges: list[tuple[int, int]] = []
    for cp in range(0x300, 0x10000):
        if unicodedata.category(chr(cp)).startswith("M"):
            if ranges and ranges[-1][1] == cp - 1:
                ranges[-1] = (ranges[-1][0], cp)
            else:
                ranges.append((cp, cp))
    return "".join(f"\\u{lo:04x}-\\u{hi:04x}" for lo, hi in ranges)


# A word is a run of letters, digits and combining marks in any script (so Devanagari vowel
# signs stay inside their word), with inner apostrophes kept ("o'clock").
_CHAR = rf"[^\W_]|[{_mark_ranges()}]"
_WORD = re.compile(rf"(?:{_CHAR})+(?:'(?:{_CHAR})+)*")
_SPACE = re.compile(r"\s+")
_SUPERSCRIPTS = str.maketrans({"\u00b9": "^1", "\u00b2": "^2", "\u00b3": "^3"})
_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:/\d+)?")

# A short list of English function words; overlap counts ignore them.
_STOP_TEXT = """
a an the of to in on at by for with from and or but not no is are was were be been being it its
this that these those which what who whom whose when where why how as than then so if do does did
has have had can could will would shall should may might must i you he she we they them his her
their our your my me us there here
"""
STOPWORDS = frozenset(_STOP_TEXT.split())


def normalise(text: str) -> str:
    """Case-folded, NFKC-normalised text with collapsed whitespace and no trailing period."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _SPACE.sub(" ", text).strip()
    return text.rstrip(" .")


def words(text: str) -> list[str]:
    return _WORD.findall(normalise(text))


def content_words(text: str) -> set[str]:
    return {w for w in words(text) if w not in STOPWORDS}


def char_ngrams(text: str, n: int = 3) -> set[str]:
    padded = f" {normalise(text)} "
    if len(padded) <= n:
        return {padded}
    return {padded[i : i + n] for i in range(len(padded) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def parse_number(text: str) -> float | None:
    """The value of an option that is a single number, with optional unit words after it.

    Accepts integers, decimals, thousands separators, simple fractions and a
    trailing percent sign or unit (``"3.5 kg"``, ``"1,200"``, ``"3/8"``). Returns
    ``None`` when the option holds no number or more than one.
    """
    # Superscripts become "^n" before NFKC would turn them into plain digits; an exponent on a
    # unit ("m²") is dropped, a power of the number itself ("10²") is not read as a number.
    s = normalise(text.translate(_SUPERSCRIPTS)).replace("−", "-").lstrip("$€£")
    if re.search(r"\d\^", s):
        return None
    s = re.sub(r"\^\d+", "", s)
    found = [m for m in _NUMBER.finditer(s) if m.group(0) and any(c.isdigit() for c in m.group(0))]
    if len(found) != 1:
        return None
    match = found[0]
    if match.start() != 0:
        return None
    rest = s[match.end() :].strip()
    if rest and not re.fullmatch(r"%|[a-zµμ°/ ]{1,12}", rest):
        return None
    token = match.group(0).replace(",", "")
    if "/" in token:
        num, den = token.split("/")
        if not num or float(den) == 0:
            return None
        return float(num) / float(den)
    try:
        return float(token)
    except ValueError:
        return None
