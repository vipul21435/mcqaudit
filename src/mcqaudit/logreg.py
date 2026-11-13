"""Multinomial logistic regression on sparse inputs, fitted with L-BFGS, on numpy only.

The design matrix is given as coordinate triplets (row, column, value). The
model minimises

    (1/n) sum_i -log softmax(x_i W + b)_{y_i}  +  (alpha / 2) ||W||^2

with the intercept ``b`` unpenalised. ``alpha = 1 / (C n)`` gives the same
optimum as scikit-learn's ``LogisticRegression(C=C)`` with the multinomial
loss. Classes can be masked per row (an item with three options cannot have
its answer at D); masked logits are set to minus infinity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class SparseRows:
    """A sparse matrix in coordinate form with ``n_rows`` rows and ``n_cols`` columns."""

    rows: np.ndarray
    cols: np.ndarray
    vals: np.ndarray
    n_rows: int
    n_cols: int

    @classmethod
    def from_dense(cls, x: np.ndarray) -> SparseRows:
        r, c = np.nonzero(x)
        return cls(r, c, x[r, c].astype(float), x.shape[0], x.shape[1])

    def take(self, index: np.ndarray) -> SparseRows:
        """The rows in ``index``, renumbered from 0 in that order."""
        index = np.asarray(index)
        new_id = np.full(self.n_rows, -1)
        new_id[index] = np.arange(len(index))
        keep = new_id[self.rows] >= 0
        return SparseRows(
            new_id[self.rows[keep]], self.cols[keep], self.vals[keep], len(index), self.n_cols
        )

    def matmul(self, w: np.ndarray) -> np.ndarray:
        """X @ w for a dense (n_cols, k) matrix ``w``."""
        out = np.empty((self.n_rows, w.shape[1]))
        for j in range(w.shape[1]):
            out[:, j] = np.bincount(
                self.rows, weights=self.vals * w[self.cols, j], minlength=self.n_rows
            )
        return out

    def rmatmul(self, r: np.ndarray) -> np.ndarray:
        """X.T @ r for a dense (n_rows, k) matrix ``r``."""
        out = np.empty((self.n_cols, r.shape[1]))
        for j in range(r.shape[1]):
            out[:, j] = np.bincount(
                self.cols, weights=self.vals * r[self.rows, j], minlength=self.n_cols
            )
        return out


def _log_softmax(z: np.ndarray) -> np.ndarray:
    top = np.max(z, axis=1, keepdims=True)
    shifted = z - top
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


def lbfgs(
    fun: Callable[[np.ndarray], tuple[float, np.ndarray]],
    x0: np.ndarray,
    *,
    max_iter: int = 500,
    memory: int = 10,
    gtol: float = 1e-6,
    ftol: float = 1e-10,
) -> tuple[np.ndarray, int]:
    """Minimise ``fun`` (returning value and gradient) by L-BFGS with Armijo backtracking.

    Returns the minimiser and the number of iterations used.
    """
    x = x0.astype(float).copy()
    f, g = fun(x)
    s_hist: list[np.ndarray] = []
    y_hist: list[np.ndarray] = []
    for it in range(1, max_iter + 1):
        if np.max(np.abs(g)) < gtol:
            return x, it - 1
        q = g.copy()
        alphas = []
        for s, y in zip(reversed(s_hist), reversed(y_hist), strict=True):
            a = (s @ q) / (y @ s)
            alphas.append(a)
            q -= a * y
        if y_hist:
            q *= (s_hist[-1] @ y_hist[-1]) / (y_hist[-1] @ y_hist[-1])
        for (s, y), a in zip(zip(s_hist, y_hist, strict=True), reversed(alphas), strict=True):
            b = (y @ q) / (y @ s)
            q += (a - b) * s
        direction = -q
        slope = g @ direction
        if slope >= 0:  # not a descent direction: restart from the gradient
            s_hist.clear()
            y_hist.clear()
            direction = -g
            slope = -(g @ g)
        step = 1.0 if s_hist else min(1.0, 1.0 / max(np.sum(np.abs(g)), 1e-12))
        while True:
            x_new = x + step * direction
            f_new, g_new = fun(x_new)
            if f_new <= f + 1e-4 * step * slope or step < 1e-12:
                break
            step *= 0.5
        s_vec, y_vec = x_new - x, g_new - g
        if s_vec @ y_vec > 1e-12:
            s_hist.append(s_vec)
            y_hist.append(y_vec)
            if len(s_hist) > memory:
                s_hist.pop(0)
                y_hist.pop(0)
        converged = abs(f - f_new) <= ftol * max(1.0, abs(f))
        x, f, g = x_new, f_new, g_new
        if converged:
            return x, it
    return x, max_iter


class SoftmaxRegression:
    """L2-penalised multinomial logistic regression with an unpenalised intercept."""

    def __init__(self, n_classes: int, *, C: float = 1.0, max_iter: int = 500, tol: float = 1e-6):
        if n_classes < 2:
            raise ValueError("need at least two classes")
        self.n_classes = n_classes
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.n_iter_ = 0

    def _masked(self, z: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        if mask is None:
            return z
        return np.where(mask, z, -np.inf)

    def fit(
        self, x: SparseRows | np.ndarray, y: np.ndarray, mask: np.ndarray | None = None
    ) -> SoftmaxRegression:
        xs = x if isinstance(x, SparseRows) else SparseRows.from_dense(np.asarray(x, float))
        y = np.asarray(y, dtype=int)
        n, d, k = xs.n_rows, xs.n_cols, self.n_classes
        if len(y) != n:
            raise ValueError("x and y differ in length")
        if mask is not None and not np.all(mask[np.arange(n), y]):
            raise ValueError("a label falls on a masked class")
        onehot = np.zeros((n, k))
        onehot[np.arange(n), y] = 1.0
        alpha = 1.0 / (self.C * n)

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            w = theta[: d * k].reshape(d, k)
            b = theta[d * k :]
            z = self._masked(xs.matmul(w) + b, mask)
            logp = _log_softmax(z)
            loss = -np.sum(np.where(onehot > 0, logp, 0.0)) / n + 0.5 * alpha * np.sum(w * w)
            resid = (np.exp(logp) - onehot) / n
            grad_w = xs.rmatmul(resid) + alpha * w
            grad_b = resid.sum(axis=0)
            return float(loss), np.concatenate([grad_w.ravel(), grad_b])

        theta, self.n_iter_ = lbfgs(
            objective, np.zeros(d * k + k), max_iter=self.max_iter, gtol=self.tol
        )
        self.coef_ = theta[: d * k].reshape(d, k)
        self.intercept_ = theta[d * k :]
        return self

    def decision_function(
        self, x: SparseRows | np.ndarray, mask: np.ndarray | None = None
    ) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("fit the model first")
        xs = x if isinstance(x, SparseRows) else SparseRows.from_dense(np.asarray(x, float))
        return self._masked(xs.matmul(self.coef_) + self.intercept_, mask)

    def predict_proba(
        self, x: SparseRows | np.ndarray, mask: np.ndarray | None = None
    ) -> np.ndarray:
        return np.exp(_log_softmax(self.decision_function(x, mask)))

    def predict(self, x: SparseRows | np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        return np.argmax(self.decision_function(x, mask), axis=1)
