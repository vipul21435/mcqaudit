"""Markdown and JSON renderings of audit results."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from .cues import CueResult
from .data import LETTERS
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
    header = "| options | n | " + " | ".join(LETTERS[:width]) + " | chi-square | df | p | method |"
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
            share = stratum.shares[LETTERS.index(top)]
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
        "| p (one-sided) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.covered == 0:
            lines.append(f"| {r.name} | 0 | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {r.name} | {r.covered} | {_pct(r.accuracy)} | {_pct(r.ci_low)} to "
            f"{_pct(r.ci_high)} | {_pct(r.chance)} | {100 * r.lift:+.1f} pt | {_z(r.z)} | "
            f"{_p(r.p_value)} |"
        )
    lines += [
        "",
        "Ties are scored by expected credit under random tie-breaking; a heuristic that",
        "ties every option, or does not apply, abstains on that item.",
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
