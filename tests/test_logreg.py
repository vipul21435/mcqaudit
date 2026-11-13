import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mcqaudit.logreg import SoftmaxRegression, SparseRows, lbfgs


def _problem(seed=0, n=300, d=12, k=4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)) * (rng.random((n, d)) < 0.5)
    w = rng.normal(size=(d, k))
    y = np.argmax(x @ w + rng.gumbel(size=(n, k)), axis=1)
    return x, y


@pytest.mark.parametrize("C", [0.1, 1.0, 10.0])
def test_matches_sklearn(C):
    x, y = _problem()
    ours = SoftmaxRegression(4, C=C, tol=1e-9, max_iter=2000).fit(x, y)
    ref = LogisticRegression(C=C, tol=1e-10, max_iter=5000).fit(x, y)
    assert np.allclose(ours.predict_proba(x), ref.predict_proba(x), atol=2e-5)
    assert np.allclose(ours.coef_, ref.coef_.T, atol=2e-3)
    assert (ours.predict(x) == ref.predict(x)).mean() > 0.99


def test_sparse_and_dense_inputs_agree():
    x, y = _problem(seed=1, n=120, d=8, k=3)
    a = SoftmaxRegression(3).fit(x, y)
    b = SoftmaxRegression(3).fit(SparseRows.from_dense(x), y)
    assert np.allclose(a.coef_, b.coef_)


def test_take_and_matmul():
    x, _ = _problem(seed=2, n=20, d=5, k=3)
    xs = SparseRows.from_dense(x)
    w = np.arange(15, dtype=float).reshape(5, 3)
    assert np.allclose(xs.matmul(w), x @ w)
    r = np.arange(60, dtype=float).reshape(20, 3)
    assert np.allclose(xs.rmatmul(r), x.T @ r)
    idx = np.array([4, 0, 7])
    assert np.allclose(xs.take(idx).matmul(w), x[idx] @ w)


def test_mask_excludes_classes():
    x, y = _problem(seed=3, n=200, d=6, k=4)
    mask = np.ones((200, 4), bool)
    mask[:50, 3] = False
    y = y.copy()
    y[:50] = np.minimum(y[:50], 2)
    model = SoftmaxRegression(4).fit(x, y, mask)
    p = model.predict_proba(x, mask)
    assert np.all(p[:50, 3] == 0)
    assert np.allclose(p.sum(axis=1), 1)
    with pytest.raises(ValueError, match="masked"):
        SoftmaxRegression(4).fit(x[:1], np.array([3]), mask[:1])


def test_lbfgs_on_a_quadratic():
    a = np.diag([1.0, 10.0, 100.0])
    target = np.array([1.0, -2.0, 3.0])

    def f(v):
        d = v - target
        return float(0.5 * d @ a @ d), a @ d

    x, iters = lbfgs(f, np.zeros(3), gtol=1e-10)
    assert np.allclose(x, target, atol=1e-7)
    assert iters < 50


def test_errors():
    with pytest.raises(ValueError):
        SoftmaxRegression(1)
    with pytest.raises(RuntimeError):
        SoftmaxRegression(2).predict(np.zeros((1, 2)))
    with pytest.raises(ValueError):
        SoftmaxRegression(2).fit(np.zeros((2, 2)), np.array([0]))
