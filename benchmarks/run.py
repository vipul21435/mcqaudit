"""Run every mcqaudit check on the public benchmarks fetched by fetch.sh.

Usage: ``bash benchmarks/fetch.sh data`` then
``python benchmarks/run.py data --json results.json > benchmarks/results.md``.

Prints a Markdown summary (numbers only, no dataset rows or ids) and writes the
full results as JSON next to it when ``--json PATH`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from mcqaudit.choices import choices_only
from mcqaudit.cues import evaluate_all
from mcqaudit.data import load, position_label
from mcqaudit.hygiene import hygiene, split_overlap
from mcqaudit.keyaudit import audit_key

# name, evaluated file, file checked for overlap
BENCHMARKS = [
    ("MMLU test", "mmlu/test.parquet", "mmlu/validation.parquet"),
    ("ARC-Challenge test", "arc/challenge-test.parquet", "arc/challenge-train.parquet"),
    ("CommonsenseQA validation", "csqa/validation.parquet", "csqa/train.parquet"),
    ("MedMCQA validation", "medmcqa/validation.parquet", "medmcqa/train.parquet"),
]

MIN_COVERAGE = 30  # a heuristic must apply to this many items to be called the best


def pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def fmt_p(p: float | None) -> str:
    if p is None:
        return "n/a"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def run_one(name: str, path: Path, against: Path | None) -> dict:
    t0 = time.time()
    ds = load(path)
    items = ds.items
    audit = audit_key(items)
    cues = evaluate_all(items)
    best = max((c for c in cues if c.covered >= MIN_COVERAGE), key=lambda c: c.ci_low - c.chance)
    co = choices_only(items)
    hy = hygiene(items)
    ov = None
    if against is not None and against.exists():
        ov = split_overlap(load(against).items, items)
    main_stratum = max(audit.positions.strata, key=lambda s: s.n)
    shares = ", ".join(f"{position_label(j)} {pct(s)}" for j, s in enumerate(main_stratum.shares))
    return {
        "name": name,
        "file": str(path.relative_to(path.parent.parent)),
        "n": len(items),
        "subjects": len({it.subject for it in items if it.subject}),
        "positions": {
            "shares_main_stratum": shares,
            "main_stratum_options": main_stratum.n_options,
            "statistic": audit.positions.statistic,
            "df": audit.positions.df,
            "p": audit.positions.p_value,
            "method": audit.positions.method,
            "subjects_p_holm_below_0.05": sum(s.p_holm < 0.05 for s in audit.subjects),
        },
        "runs": {"z": audit.runs.z, "p": audit.runs.p_value},
        "best_cue": asdict(best) | {"lift": best.lift},
        "cues": [asdict(c) | {"lift": c.lift} for c in cues],
        "choices_only": asdict(co) | {"lift": co.lift},
        "hygiene": asdict(hy) | {"examples": None},
        "overlap": None if ov is None else asdict(ov) | {"examples": None, "against": against.name},
        "seconds": round(time.time() - t0, 1),
    }


def markdown(results: list[dict]) -> str:
    out = ["# mcqaudit on public benchmarks", ""]
    out += [
        "| benchmark | items | answer positions | position test p | runs z (p) "
        "| best heuristic | its accuracy (95 % CI) | chance | p (Holm) "
        "| choices-only accuracy (95 % CI) | majority position |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        b, c, pos = r["best_cue"], r["choices_only"], r["positions"]
        runs = r["runs"]
        runs_txt = "n/a" if runs["z"] is None else f"{runs['z']:+.2f} ({fmt_p(runs['p'])})"
        out.append(
            f"| {r['name']} | {r['n']} | {pos['shares_main_stratum']} | {fmt_p(pos['p'])} "
            f"| {runs_txt} | {b['name']} (covers {b['covered']}) | {pct(b['accuracy'])} "
            f"({pct(b['ci_low'])} to {pct(b['ci_high'])}) | {pct(b['chance'])} "
            f"| {fmt_p(b['p_holm'])} | {pct(c['accuracy'])} "
            f"({pct(c['ci_low'])} to {pct(c['ci_high'])}), chance "
            f"{pct(c['chance'])}, {c['folds']} | {pct(c['majority_accuracy'])} |"
        )
    out += ["", "## Hygiene", ""]
    out += [
        "| benchmark | duplicate options | empty options | duplicated questions (items) "
        "| repeated item groups | conflicting duplicates | answer in question (only answer) "
        "| option counts | overlap: exact / near (against) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        h, o = r["hygiene"], r["overlap"]
        counts = ", ".join(f"{k}: {v}" for k, v in h["option_counts"].items())
        ov = "n/a" if o is None else f"{o['exact']} / {o['near']} ({o['against']})"
        out.append(
            f"| {r['name']} | {h['duplicate_options']} | {h['empty_options']} "
            f"| {h['duplicate_question_groups']} ({h['duplicate_question_items']}) "
            f"| {h['duplicate_item_groups']} "
            f"| {h['conflicting_duplicates']} | {h['answer_in_question']} "
            f"({h['answer_only_in_question']}) | {counts} | {ov} |"
        )
    out += ["", "## Every heuristic", ""]
    for r in results:
        out += [
            f"### {r['name']}",
            "",
            "| heuristic | covered | accuracy | 95 % CI | chance | p | p (Holm) |",
            "|---|---|---|---|---|---|---|",
        ]
        for c in r["cues"]:
            if c["covered"] == 0:
                continue
            out.append(
                f"| {c['name']} | {c['covered']} | {pct(c['accuracy'])} | {pct(c['ci_low'])} to "
                f"{pct(c['ci_high'])} | {pct(c['chance'])} | {fmt_p(c['p_value'])} "
                f"| {fmt_p(c['p_holm'])} |"
            )
        out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", help="directory filled by benchmarks/fetch.sh")
    parser.add_argument("--json", help="also write the full results here")
    args = parser.parse_args()
    root = Path(args.data)
    results = []
    for name, rel, other in BENCHMARKS:
        path = root / rel
        if not path.exists():
            print(f"skip {name}: {path} missing", file=sys.stderr)
            continue
        print(f"running {name} ...", file=sys.stderr)
        results.append(run_one(name, path, root / other if other else None))
    sys.stdout.write(markdown(results) + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
