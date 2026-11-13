"""Run every check on the small example file through the Python API."""

from pathlib import Path

from mcqaudit.choices import choices_only
from mcqaudit.cues import evaluate_all
from mcqaudit.data import load
from mcqaudit.hygiene import hygiene
from mcqaudit.keyaudit import audit_key
from mcqaudit.shuffle import balance_positions

here = Path(__file__).resolve().parent
ds = load(here / "sample.jsonl")
print(f"{len(ds)} items, layouts {ds.layouts}")

audit = audit_key(ds.items)
stratum = audit.positions.strata[0]
print(f"answer counts {stratum.counts}, {stratum.method} p = {audit.positions.p_value:.4f}")
print(f"runs test z = {audit.runs.z:+.2f}, p = {audit.runs.p_value:.4f}")

for cue in evaluate_all(ds.items)[:3]:
    print(
        f"{cue.name:24s} {cue.accuracy:6.1%} on {cue.covered:2d} items "
        f"(95 % CI {cue.ci_low:.1%} to {cue.ci_high:.1%}, chance {cue.chance:.1%})"
    )

co = choices_only(ds.items, n_folds=2)
print(f"choices-only accuracy {co.accuracy:.1%} ({co.folds}), chance {co.chance:.1%}")

rep = hygiene(ds.items)
print(f"duplicate options {rep.duplicate_options}, empty options {rep.empty_options}")

balanced = balance_positions(ds.items, seed=0)
after = audit_key(balanced).positions
print(f"after balancing: counts {after.strata[0].counts}, p = {after.p_value:.4f}")
