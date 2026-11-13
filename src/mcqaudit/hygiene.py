"""Item hygiene: defects that make an item ambiguous, trivial or duplicated.

All comparisons use :func:`mcqaudit.text.normalise` (case-folded, NFKC,
collapsed whitespace, no trailing period). Near-duplicate questions across two
files are found by Jaccard similarity of word 3-gram shingles (word sets for
questions under three words), with candidates drawn from an inverted index.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from .data import Item
from .text import jaccard, normalise, words

MIN_ANSWER_CHARS = 3


@dataclass
class HygieneReport:
    n_items: int
    duplicate_options: int
    empty_options: int
    duplicate_question_groups: int
    duplicate_question_items: int
    duplicate_item_groups: int
    duplicate_questions_all_options_differ: int
    conflicting_duplicates: int
    answer_in_question: int
    answer_only_in_question: int
    option_counts: dict[int, int]
    off_mode_option_count: int
    examples: dict[str, list[str]] = field(default_factory=dict)


def _contains(haystack: str, needle: str) -> bool:
    if len(needle) < MIN_ANSWER_CHARS:
        return False
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack) is not None


def hygiene(items: Sequence[Item], *, max_examples: int = 5) -> HygieneReport:
    """Count duplicate, empty and leaking options and duplicated questions."""
    examples: dict[str, list[str]] = defaultdict(list)

    def note(kind: str, item: Item) -> None:
        if len(examples[kind]) < max_examples:
            examples[kind].append(item.id)

    dup_opts = empty = in_q = only_in_q = 0
    by_question: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        opts = [normalise(o) for o in item.options]
        if len(set(opts)) < len(opts):
            dup_opts += 1
            note("duplicate_options", item)
        if any(not o for o in opts):
            empty += 1
            note("empty_options", item)
        question = normalise(item.question)
        if question:
            by_question[question].append(item)
        answer = opts[item.answer]
        if _contains(question, answer):
            in_q += 1
            note("answer_in_question", item)
            others = [o for i, o in enumerate(opts) if i != item.answer]
            if not any(_contains(question, o) for o in others):
                only_in_q += 1

    groups = [g for g in by_question.values() if len(g) > 1]
    conflicting = repeated = all_differ = 0
    for group in groups:
        note("duplicate_questions", group[0])
        by_options: dict[frozenset[str], list[str]] = defaultdict(list)
        for it in group:
            key = frozenset(normalise(o) for o in it.options)
            by_options[key].append(normalise(it.answer_text))
        repeated += sum(1 for answers in by_options.values() if len(answers) > 1)
        all_differ += all(len(answers) == 1 for answers in by_options.values())
        if any(len(set(answers)) > 1 for answers in by_options.values()):
            conflicting += 1
            note("conflicting_duplicates", group[0])

    counts = Counter(item.n_options for item in items)
    mode = counts.most_common(1)[0][0] if counts else 0
    off_mode = sum(c for k, c in counts.items() if k != mode)
    return HygieneReport(
        n_items=len(items),
        duplicate_options=dup_opts,
        empty_options=empty,
        duplicate_question_groups=len(groups),
        duplicate_question_items=sum(len(g) for g in groups),
        duplicate_item_groups=repeated,
        duplicate_questions_all_options_differ=all_differ,
        conflicting_duplicates=conflicting,
        answer_in_question=in_q,
        answer_only_in_question=only_in_q,
        option_counts=dict(sorted(counts.items())),
        off_mode_option_count=off_mode,
        examples=dict(examples),
    )


def shingles(text: str, n: int = 3) -> set[str]:
    toks = words(text)
    if len(toks) < n:
        return set(toks)
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


@dataclass
class OverlapReport:
    n_a: int
    n_b: int
    threshold: float
    exact: int
    near: int
    examples: list[tuple[str, str, float]] = field(default_factory=list)


def split_overlap(
    a: Sequence[Item],
    b: Sequence[Item],
    *,
    threshold: float = 0.8,
    max_postings: int = 200,
    max_examples: int = 5,
) -> OverlapReport:
    """Items of ``b`` whose question matches one in ``a`` exactly or nearly.

    ``near`` counts items of ``b`` with a best Jaccard similarity at or above
    ``threshold`` (exact matches included). Shingles that occur in more than
    ``max_postings`` questions of ``a`` are not used to find candidates.
    """
    exact_index = {normalise(it.question) for it in a if normalise(it.question)}
    a_sets = [shingles(it.question) for it in a]
    index: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(a_sets):
        for sh in s:
            index[sh].append(i)
    exact = near = 0
    examples: list[tuple[str, str, float]] = []
    for item in b:
        q = normalise(item.question)
        if q and q in exact_index:
            exact += 1
        s = shingles(item.question)
        if not s:
            continue
        candidates: Counter[int] = Counter()
        for sh in s:
            posting = index.get(sh)
            if posting and len(posting) <= max_postings:
                candidates.update(posting)
        best, best_i = 0.0, -1
        for i, _shared in candidates.most_common(20):
            sim = jaccard(s, a_sets[i])
            if sim > best:
                best, best_i = sim, i
        if best >= threshold:
            near += 1
            if len(examples) < max_examples:
                examples.append((a[best_i].id, item.id, round(best, 3)))
    return OverlapReport(len(a), len(b), threshold, exact, near, examples)
