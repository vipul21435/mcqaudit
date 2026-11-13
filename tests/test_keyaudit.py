import itertools
import random
import statistics

import pytest
from scipy import stats as sps

from mcqaudit.data import Item
from mcqaudit.keyaudit import audit_key, position_summary, position_test, runs_test


def make_items(answers, n_options=4, subject=None):
    return [
        Item(f"q{i}", f"question {i}", tuple(f"opt{j}" for j in range(n_options)), a, subject)
        for i, a in enumerate(answers)
    ]


def test_position_test_chi_square_matches_scipy():
    rng = random.Random(3)
    answers = [rng.choice([0, 0, 1, 2, 3]) for _ in range(400)]
    res = position_test(answers, 4, method="chi-square")
    ref = sps.chisquare(res.counts)
    assert res.statistic == pytest.approx(ref.statistic)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert res.top_position == "A"
    assert sum(res.shares) == pytest.approx(1.0)


def test_position_test_auto_uses_exact_when_small():
    res = position_test([0, 0, 0, 1, 2, 0, 0, 3], 4)
    assert res.method == "exact multinomial"
    big = position_test([i % 4 for i in range(2000)], 4)
    assert big.method == "chi-square"
    assert big.p_value == pytest.approx(1.0)


def test_position_test_input_checks():
    with pytest.raises(ValueError):
        position_test([0, 5], 4)
    with pytest.raises(ValueError):
        position_test([0], 1)
    with pytest.raises(ValueError):
        position_test([0, 1], 2, method="bogus")
    assert position_test([], 4).p_value == 1.0


def test_mixed_option_counts_are_stratified():
    items = make_items([0, 1, 2, 3] * 10, 4) + make_items([0, 0, 0, 0, 1, 2] * 5, 3)
    summary = position_summary(items, method="chi-square")
    assert [s.n_options for s in summary.strata] == [3, 4]
    assert summary.df == 2 + 3
    assert summary.statistic == pytest.approx(sum(s.statistic for s in summary.strata))
    assert summary.p_value == pytest.approx(sps.chi2.sf(summary.statistic, 5), rel=1e-9)
    assert summary.p_value < 0.05


def _distinct_permutations(seq):
    return set(itertools.permutations(seq))


@pytest.mark.parametrize("multiset", [(0, 0, 1, 1, 1), (0, 1, 2, 2), (0, 0, 1, 2, 2, 3), (1, 1)])
def test_runs_moments_match_full_enumeration(multiset):
    perms = _distinct_permutations(multiset)
    runs = [1 + sum(a != b for a, b in itertools.pairwise(p)) for p in perms]
    res = runs_test(list(multiset))
    assert res.expected_runs == pytest.approx(statistics.fmean(runs))
    assert res.sd_runs**2 == pytest.approx(statistics.pvariance(runs), abs=1e-12)


def test_runs_two_categories_is_wald_wolfowitz():
    seq = [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1]
    n1, n2 = seq.count(0), seq.count(1)
    n = n1 + n2
    mean = 2 * n1 * n2 / n + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n * n * (n - 1))
    res = runs_test(seq)
    assert res.expected_runs == pytest.approx(mean)
    assert res.sd_runs**2 == pytest.approx(var)


def test_runs_flags_alternating_and_clustered_keys():
    alternating = [0, 1, 2, 3] * 25
    clustered = sorted(alternating)
    hi = runs_test(alternating)
    lo = runs_test(clustered)
    assert hi.z > 3 and hi.p_value < 1e-3
    assert lo.z < -3 and lo.p_value < 1e-3
    assert lo.repeats == 96


def test_runs_degenerate_cases():
    assert runs_test([]).p_value is None
    assert runs_test([2]).p_value is None
    assert runs_test([1, 1, 1]).p_value is None


def test_planted_position_bias_is_found():
    rng = random.Random(11)
    answers = [rng.choices([0, 1, 2, 3], weights=[0.25, 0.25, 0.35, 0.15])[0] for _ in range(800)]
    audit = audit_key(make_items(answers))
    assert audit.positions.p_value < 1e-4
    assert audit.positions.strata[0].top_position == "C"


def test_false_positive_rate_under_the_null():
    rng = random.Random(2026)
    trials, rejections_pos, rejections_runs = 400, 0, 0
    for _ in range(trials):
        answers = [rng.randrange(4) for _ in range(120)]
        audit = audit_key(make_items(answers))
        rejections_pos += audit.positions.p_value < 0.05
        rejections_runs += audit.runs.p_value < 0.05
    # Binomial(400, 0.05) has mean 20 and sd about 4.4.
    assert 6 <= rejections_pos <= 36
    assert 6 <= rejections_runs <= 36


def test_exact_false_positive_rate_is_at_most_nominal():
    rng = random.Random(7)
    trials = 400
    rejections = sum(
        position_test([rng.randrange(4) for _ in range(30)], 4, method="exact").p_value < 0.05
        for _ in range(trials)
    )
    assert rejections <= 36


def test_by_subject_with_holm():
    items = make_items([0] * 40, subject="biased") + make_items(
        [0, 1, 2, 3] * 10, subject="balanced"
    )
    audit = audit_key(items)
    by_name = {s.subject: s for s in audit.subjects}
    assert set(by_name) == {"balanced", "biased"}
    assert by_name["biased"].positions.p_value < 1e-10
    assert by_name["biased"].p_holm <= 2 * by_name["biased"].positions.p_value + 1e-15
    assert by_name["balanced"].p_holm == pytest.approx(1.0)
    assert audit_key(items, by_subject=False).subjects == []
