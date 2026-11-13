import itertools
import math

import pytest
from scipy import stats as sps

from mcqaudit.stats import (
    chi2_sf,
    chi_square_statistic,
    gamma_q,
    holm_adjust,
    multinomial_exact_p,
    normal_two_sided_p,
    normal_upper_p,
    wilson_interval,
)


@pytest.mark.parametrize("df", [1, 2, 3, 4, 7, 10, 30, 99])
@pytest.mark.parametrize("x", [0.001, 0.1, 0.5, 1.0, 2.5, 5.0, 11.3, 25.0, 60.0, 200.0])
def test_chi2_sf_matches_scipy(x, df):
    ours = chi2_sf(x, df)
    ref = sps.chi2.sf(x, df)
    assert ours == pytest.approx(ref, rel=1e-9, abs=1e-300)


def test_chi2_sf_edges():
    assert chi2_sf(0.0, 3) == 1.0
    assert chi2_sf(-1.0, 3) == 1.0
    with pytest.raises(ValueError):
        chi2_sf(1.0, 0)
    with pytest.raises(ValueError):
        gamma_q(0.0, 1.0)


@pytest.mark.parametrize("a", [0.5, 1.0, 3.7, 12.0])
@pytest.mark.parametrize("x", [0.2, 3.0, 15.0])
def test_gamma_q_matches_scipy(a, x):
    from scipy.special import gammaincc

    assert gamma_q(a, x) == pytest.approx(gammaincc(a, x), rel=1e-10)


def test_normal_p_values():
    assert normal_two_sided_p(1.959963984540054) == pytest.approx(0.05, rel=1e-9)
    assert normal_upper_p(0.0) == pytest.approx(0.5)
    assert normal_upper_p(2.0) == pytest.approx(sps.norm.sf(2.0), rel=1e-12)


@pytest.mark.parametrize("k,n", [(0, 10), (3, 10), (10, 10), (17, 40), (250, 1000)])
@pytest.mark.parametrize("conf", [0.9, 0.95, 0.99])
def test_wilson_matches_scipy(k, n, conf):
    ref = sps.binomtest(k, n).proportion_ci(confidence_level=conf, method="wilson")
    lo, hi = wilson_interval(k, n, conf)
    assert lo == pytest.approx(ref.low, abs=1e-12)
    assert hi == pytest.approx(ref.high, abs=1e-12)


def test_wilson_hand_worked():
    # 8 of 10 at 95 %: centre (0.8 + 1.9208/10) / 1.38415 = 0.7167.
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4)
    assert hi == pytest.approx(0.9433, abs=1e-4)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_chi_square_statistic():
    assert chi_square_statistic([10, 20, 30, 40], [25, 25, 25, 25]) == pytest.approx(20.0)
    ref = sps.chisquare([10, 20, 30, 40])
    assert chi_square_statistic([10, 20, 30, 40], [25] * 4) == pytest.approx(ref.statistic)


@pytest.mark.parametrize("x,n", [(0, 5), (2, 9), (7, 20), (13, 20), (31, 60)])
def test_exact_with_two_cells_is_binomial_test(x, n):
    ours = multinomial_exact_p([x, n - x])
    ref = sps.binomtest(x, n, 0.5).pvalue
    assert ours == pytest.approx(ref, rel=1e-9)


def test_exact_two_cells_unequal_probs():
    ours = multinomial_exact_p([3, 17], [0.3, 0.7])
    ref = sps.binomtest(3, 20, 0.3).pvalue
    assert ours == pytest.approx(ref, rel=1e-9)


def _brute_force_exact(counts, probs):
    n, k = sum(counts), len(counts)
    obs = sps.multinomial.pmf(counts, n, probs)
    total = 0.0
    for table in itertools.product(range(n + 1), repeat=k):
        if sum(table) != n:
            continue
        p = sps.multinomial.pmf(table, n, probs)
        if p <= obs * (1 + 1e-7):
            total += p
    return total


@pytest.mark.parametrize(
    "counts", [[3, 1, 0, 2], [5, 0, 0, 0], [2, 2, 2, 1], [1, 4, 1, 6], [0, 3, 5]]
)
def test_exact_matches_brute_force(counts):
    probs = [1.0 / len(counts)] * len(counts)
    assert multinomial_exact_p(counts) == pytest.approx(_brute_force_exact(counts, probs), rel=1e-9)


def test_exact_balanced_table_has_p_one():
    assert multinomial_exact_p([5, 5, 5, 5]) == pytest.approx(1.0)
    assert multinomial_exact_p([0, 0, 0]) == 1.0


def test_exact_refuses_large_tables():
    with pytest.raises(ValueError, match="max_outcomes"):
        multinomial_exact_p([500, 500, 500, 500], max_outcomes=1000)


def test_exact_input_checks():
    with pytest.raises(ValueError):
        multinomial_exact_p([3])
    with pytest.raises(ValueError):
        multinomial_exact_p([1, -1])
    with pytest.raises(ValueError):
        multinomial_exact_p([1, 2], [0.0, 1.0])


def test_holm_adjust():
    p = [0.01, 0.04, 0.03, 0.005]
    adj = holm_adjust(p)
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1 -> max(0.06, 0.04)=0.06
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm_adjust([]) == []
    assert all(0 <= a <= 1 for a in holm_adjust([0.9, 0.8, 0.7]))
    assert math.isclose(max(holm_adjust([0.9, 0.8, 0.7])), 1.0)


def test_credit_upper_p_closed_forms():
    from mcqaudit.stats import credit_upper_p

    # 8 of 8 single picks right with four options: 0.25 ** 8.
    assert credit_upper_p([(4, 1)] * 8, 8.0) == pytest.approx(0.25**8, rel=1e-12)
    assert credit_upper_p([(4, 1)] * 8, 0.0) == pytest.approx(1.0)
    # Binomial tail for single picks.
    assert credit_upper_p([(4, 1)] * 20, 9.0) == pytest.approx(sps.binom.sf(8, 20, 0.25))
    assert credit_upper_p([], 0.0) == 1.0
    assert credit_upper_p([(4, 1)] * 10, 3.0, max_units=5) is None


def test_credit_upper_p_matches_enumeration_with_ties():
    from mcqaudit.stats import credit_upper_p

    pairs = [(4, 2), (4, 1), (3, 2), (5, 3), (4, 2), (4, 1)]
    outcomes = [[(1.0 / c, c / k), (0.0, 1 - c / k)] for k, c in pairs]
    observed = 0.5 + 1 + 0.5 + 0 + 0.5 + 0
    total = 0.0
    for combo in itertools.product(*outcomes):
        if sum(v for v, _ in combo) >= observed - 1e-9:
            total += math.prod(q for _, q in combo)
    assert credit_upper_p(pairs, observed) == pytest.approx(total, rel=1e-12)
