"""Markdown and JSON renderings of audit results."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from .choices import ChoicesOnlyResult
from .cues import CueResult
from .data import position_label
from .hygiene import HygieneReport, OverlapReport
from .keyaudit import KeyAudit, PositionTest


def _p(p: float | None) -> str:
    if p is None:
        return "n/a"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def _z(z: float | None) -> str:
    return "n/a" if z is None else f"{z:+.2f}"


def _position_row(t: PositionTest, width: int) -> str:
    cells = []
    for j in range(width):
        if j < t.n_options:
            cells.append(f"{t.counts[j]} ({_pct(t.shares[j])})")
        else:
            cells.append("")
    return (
        f"| {t.n_options} | {t.n} | "
        + " | ".join(cells)
        + f" | {t.statistic:.2f} | {t.df} | {_p(t.p_value)} | {t.method} |"
    )


def audit_markdown(audit: KeyAudit, source: str = "") -> str:
    width = max((s.n_options for s in audit.positions.strata), default=0)
    lines = [f"# Answer-key audit{': ' + source if source else ''}", ""]
    lines.append(f"Items: {audit.n_items}")
    lines += ["", "## Answer positions", ""]
    header = (
        "| options | n | "
        + " | ".join(position_label(j) for j in range(width))
        + " | chi-square | df | p | method |"
    )
    lines += [header, "|" + "---|" * (width + 6)]
    for stratum in audit.positions.strata:
        lines.append(_position_row(stratum, width))
    if len(audit.positions.strata) > 1:
        pos = audit.positions
        lines += [
            "",
            f"Combined over option counts: chi-square {pos.statistic:.2f}, df {pos.df}, "
            f"p {_p(pos.p_value)}.",
        ]
        main = max(pos.strata, key=lambda t: t.n)
        lines.append(
            f"The largest stratum ({main.n_options} options, {main.n} items) alone has "
            f"p {_p(main.p_value)}; summing over strata is conservative when some are small."
        )
    r = audit.runs
    lines += ["", "## Order of the key (runs test, file order)", ""]
    lines += ["| n | runs | expected | z | p |", "|---|---|---|---|---|"]
    lines.append(f"| {r.n} | {r.runs} | {r.expected_runs:.1f} | {_z(r.z)} | {_p(r.p_value)} |")
    if r.z is not None:
        trend = (
            "fewer runs than chance (answers cluster)"
            if r.z < 0
            else ("more runs than chance (the key avoids repeating a position)")
        )
        lines += ["", f"z {_z(r.z)}, {trend}."]
    if audit.subjects:
        lines += ["", "## By subject", ""]
        lines += [
            "| subject | n | top | top share | p | p (Holm) | runs z |",
            "|---|---|---|---|---|---|---|",
        ]
        for s in audit.subjects:
            stratum = max(s.positions.strata, key=lambda t: t.n)
            top = stratum.top_position
            share = max(stratum.shares)
            lines.append(
                f"| {s.subject} | {s.n} | {top} | {_pct(share)} | {_p(s.positions.p_value)} | "
                f"{_p(s.p_holm)} | {_z(s.runs.z)} |"
            )
    return "\n".join(lines) + "\n"


def cues_markdown(results: Sequence[CueResult], source: str = "", confidence: float = 0.95) -> str:
    lines = [f"# Question-free heuristics{': ' + source if source else ''}", ""]
    level = f"{100 * confidence:g} %"
    lines += [
        f"| heuristic | covered | accuracy | {level} interval | chance | lift | z "
        "| p (one-sided) | p (Holm) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.covered == 0:
            lines.append(f"| {r.name} | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {r.name} | {r.covered} | {_pct(r.accuracy)} | {_pct(r.ci_low)} to "
            f"{_pct(r.ci_high)} | {_pct(r.chance)} | {100 * r.lift:+.1f} pt | {_z(r.z)} | "
            f"{_p(r.p_value)} | {_p(r.p_holm)} |"
        )
    lines += [
        "",
        "Ties are scored by expected credit under random tie-breaking; a heuristic that",
        "ties every option, or does not apply, abstains on that item. p-values are exact",
        "for heuristics covering up to 500 items and use the normal approximation above.",
        "Each interval and p-value is for one heuristic on its own; with this many heuristics,",
        "read the Holm column (adjusted over every heuristic in the table) before calling one",
        "above chance.",
    ]
    return "\n".join(lines) + "\n"


def audit_dict(audit: KeyAudit) -> dict[str, Any]:
    data = asdict(audit)
    for stratum, raw in zip(audit.positions.strata, data["positions"]["strata"], strict=True):
        raw["shares"] = stratum.shares
        raw["top_position"] = stratum.top_position
    data["runs"]["repeats"] = audit.runs.repeats
    return data


def cues_dict(results: Sequence[CueResult]) -> list[dict[str, Any]]:
    out = []
    for r in results:
        row = asdict(r)
        row["lift"] = r.lift
        out.append(row)
    return out


def to_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=False) + "\n"


def choices_markdown(res: ChoicesOnlyResult, source: str = "", confidence: float = 0.95) -> str:
    level = f"{100 * confidence:g} %"
    lines = [f"# Choices-only classifier{': ' + source if source else ''}", ""]
    lines += [
        f"Multinomial logistic regression on hashed word 1-2 grams of the options "
        f"({res.n_features} columns per position, C = {res.C:g}); the question is never read. "
        f"{res.n_folds} folds, {res.folds}.",
        "",
        f"| items | correct | accuracy | {level} interval | chance | lift | z | p (one-sided) "
        "| majority position |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {res.n_items} | {res.correct} | {_pct(res.accuracy)} | {_pct(res.ci_low)} to "
        f"{_pct(res.ci_high)} | {_pct(res.chance)} | {100 * res.lift:+.1f} pt | {_z(res.z)} | "
        f"{_p(res.p_value)} | {_pct(res.majority_accuracy)} |",
        "",
        "Majority position: accuracy of always picking the training folds' most common position.",
        f"{res.test_items_seen_in_training} held-out items had their option set in another fold; "
        "those copies were left out of training.",
    ]
    if res.folds_at_iteration_cap:
        lines += [
            "",
            f"Warning: L-BFGS stopped at its cap of {res.max_iter} iterations in "
            f"{res.folds_at_iteration_cap} of {res.n_folds} folds, short of the optimum; "
            "raise --max-iter.",
        ]
    return "\n".join(lines) + "\n"


def choices_dict(res: ChoicesOnlyResult) -> dict[str, Any]:
    row = asdict(res)
    row["lift"] = res.lift
    return row


def hygiene_markdown(
    rep: HygieneReport, overlap: OverlapReport | None = None, source: str = ""
) -> str:
    lines = [f"# Item hygiene{': ' + source if source else ''}", ""]
    counts = ", ".join(f"{k} options: {v}" for k, v in rep.option_counts.items())
    rows = [
        ("items", rep.n_items),
        ("items with duplicate options", rep.duplicate_options),
        ("items with an empty option", rep.empty_options),
        ("duplicated questions (groups)", rep.duplicate_question_groups),
        ("items in those groups", rep.duplicate_question_items),
        (
            "... groups whose items all have different options",
            rep.duplicate_questions_all_options_differ,
        ),
        ("groups of repeated items (same question, same options)", rep.duplicate_item_groups),
        ("groups giving different answers to the same options", rep.conflicting_duplicates),
        ("answer text appears in the question", rep.answer_in_question),
        ("... and no distractor does", rep.answer_only_in_question),
        ("items off the usual option count", rep.off_mode_option_count),
    ]
    lines += ["| check | count |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    lines += ["", f"Option counts: {counts}."]
    if rep.examples:
        lines += ["", "Example ids:", ""]
        for kind, ids in rep.examples.items():
            lines.append(f"- {kind.replace('_', ' ')}: {', '.join(ids)}")
    if overlap is not None:
        lines += [
            "",
            "## Overlap with the other file",
            "",
            f"Of {overlap.n_b} questions here, {overlap.exact} match a question of the other file "
            f"({overlap.n_a} questions) exactly after normalisation and {overlap.near} have word "
            f"3-gram Jaccard similarity of at least {overlap.threshold:g}.",
        ]
    return "\n".join(lines) + "\n"
