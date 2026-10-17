"""Trainable supervised models using orthogonal-polynomial features."""

import numpy as np
from scipy import optimize, special
from scipy.special import logsumexp

from .families import PolynomialFamily
from .graph import normalized_laplacian


def _ridge_readout(design: np.ndarray, targets: np.ndarray, ridge: float):
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    x = np.column_stack([np.ones(len(design)), design])
    penalty = np.diag(np.r_[0., np.full(design.shape[1], np.sqrt(ridge))])
    solution, *_ = np.linalg.lstsq(np.vstack([x, penalty]),
                                   np.concatenate([targets, np.zeros(x.shape[1])]), rcond=None)
    return solution[0], solution[1:]


class PolynomialRegressor:
    """Fit a one-dimensional response with a classical polynomial basis.

    Chebyshev and Jacobi inputs are scaled from the training range to [-1, 1].
    Hermite and Laguerre inputs retain their original scale. The fitted
    coefficients use ridge-regularized least squares (degree zero unpenalized).
    """

    def __init__(self, family: PolynomialFamily, degree: int, ridge: float = 0.0):
        if not isinstance(family, PolynomialFamily):
            raise TypeError("family must be a PolynomialFamily")
        if not isinstance(degree, (int, np.integer)) or degree < 0:
            raise ValueError("degree must be a nonnegative integer")
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        self.family, self.degree, self.ridge = family, int(degree), float(ridge)

    def _basis(self, x: np.ndarray) -> np.ndarray:
        if self.family.kind in ("chebyshev", "jacobi"):
            lo, hi = self.domain_
            x = 2 * (x - lo) / (hi - lo) - 1
        elif self.family.kind == "laguerre" and np.any(x < 0):
            raise ValueError("Laguerre inputs must be nonnegative")
        return np.column_stack([self.family.evaluate(k, x) for k in range(self.degree + 1)])

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PolynomialRegressor":
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 1 or y.shape != x.shape or len(x) < self.degree + 1:
            raise ValueError("x and y must be equal-length vectors with enough samples")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("training data must be finite")
        if self.family.kind in ("chebyshev", "jacobi"):
            lo, hi = float(x.min()), float(x.max())
            if lo == hi:
                raise ValueError("training x must span an interval")
            self.domain_ = (lo, hi)
        basis = self._basis(x)
        if self.ridge:
            penalty = np.diag(np.r_[0., np.full(self.degree, np.sqrt(self.ridge))])
            self.coef_, *_ = np.linalg.lstsq(np.vstack([basis, penalty]),
                                            np.r_[y, np.zeros(self.degree + 1)], rcond=None)
        else:
            self.coef_, *_ = np.linalg.lstsq(basis, y, rcond=None)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before predict")
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError("x must be finite")
        if self.family.kind == "laguerre" and np.any(x < 0):
            raise ValueError("Laguerre inputs must be nonnegative")
        result = self.family.evaluate_series(self.coef_,
            2 * (x - self.domain_[0]) / (self.domain_[1] - self.domain_[0]) - 1
            if self.family.kind in ("chebyshev", "jacobi") else x)
        return result


def _graph_design(adjacency, features: np.ndarray, order: int) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[1] == 0 or not np.all(np.isfinite(x)):
        raise ValueError("features must have finite shape (nodes, channels)")
    laplacian = normalized_laplacian(adjacency)
    if laplacian.shape[0] != x.shape[0]:
        raise ValueError("adjacency and features must have the same node count")
    terms = [x]
    if order:
        terms.append(laplacian @ x - x)  # T1(L - I), using lambda_max = 2
        for _ in range(2, order + 1):
            terms.append(2 * (laplacian @ terms[-1] - terms[-1]) - terms[-2])
    return np.concatenate(terms, axis=1)


def _training_rows(mask, node_count: int) -> np.ndarray:
    if mask is None:
        return np.arange(node_count)
    selected = np.asarray(mask)
    if selected.dtype == bool:
        if selected.shape != (node_count,):
            raise ValueError("boolean train_mask must have one entry per node")
        selected = np.flatnonzero(selected)
    elif selected.ndim == 1 and np.issubdtype(selected.dtype, np.integer):
        if np.any(selected < 0) or np.any(selected >= node_count):
            raise ValueError("train indices are out of bounds")
    else:
        raise ValueError("train_mask must be a boolean mask or integer indices")
    if len(selected) == 0 or len(np.unique(selected)) != len(selected):
        raise ValueError("train_mask must select distinct nodes")
    return selected


class ChebyshevGraphClassifier:
    """Train a softmax node classifier on Chebyshev graph features.

    This is a single trainable Chebyshev graph layer with a linear softmax
    readout. Labels are used only at ``train_mask`` nodes; the graph and node
    features remain visible at all nodes, as in transductive learning.
    """

    def __init__(self, order: int = 2, ridge: float = 1e-3, max_iter: int = 300):
        if not isinstance(order, (int, np.integer)) or order < 0:
            raise ValueError("order must be a nonnegative integer")
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        self.order, self.ridge, self.max_iter = int(order), float(ridge), int(max_iter)

    def fit(self, adjacency, features: np.ndarray, labels: np.ndarray, train_mask=None) -> "ChebyshevGraphClassifier":
        design = _graph_design(adjacency, features, self.order)
        labels = np.asarray(labels)
        if labels.shape != (len(design),):
            raise ValueError("labels must have one value per node")
        rows = _training_rows(train_mask, len(design))
        classes, encoded = np.unique(labels[rows], return_inverse=True)
        if len(classes) < 2:
            raise ValueError("training nodes must contain at least two classes")
        mean = design[rows].mean(axis=0)
        scale = design[rows].std(axis=0)
        scale[scale < 1e-12] = 1
        x = np.column_stack([np.ones(len(rows)), (design[rows] - mean) / scale])
        n, width = x.shape
        classes_count = len(classes)

        def objective(flat):
            weights = flat.reshape(width, classes_count)
            logits = x @ weights
            loss = (np.sum(logsumexp(logits, axis=1) - logits[np.arange(n), encoded]) / n
                    + 0.5 * self.ridge * np.sum(weights[1:]**2))
            probabilities = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
            probabilities[np.arange(n), encoded] -= 1
            gradient = x.T @ probabilities / n
            gradient[1:] += self.ridge * weights[1:]
            return loss, gradient.ravel()

        result = optimize.minimize(objective, np.zeros(width * classes_count),
                                   jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter})
        if not result.success and result.status != 1:
            raise RuntimeError(f"graph classifier training failed: {result.message}")
        self.weights_ = result.x.reshape(width, classes_count)
        self.mean_, self.scale_, self.classes_ = mean, scale, classes
        self.loss_, self.n_iter_ = float(result.fun), result.nit
        return self

    def predict_proba(self, adjacency, features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError("fit must be called before predict_proba")
        design = _graph_design(adjacency, features, self.order)
        if design.shape[1] != len(self.mean_):
            raise ValueError("prediction features must have the training channel count")
        x = np.column_stack([np.ones(len(design)), (design - self.mean_) / self.scale_])
        logits = x @ self.weights_
        return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))

    def predict(self, adjacency, features: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(adjacency, features)
        return self.classes_[np.argmax(probabilities, axis=1)]


class LaguerreForecaster:
    """Learn to predict a future sample from Laguerre-encoded history windows.

    ``train_end`` in ``fit`` is an exclusive target index. Windows whose target
    occurs at or after this index are held out, preventing target leakage.
    """

    def __init__(self, window: int, order: int, rate: float = 1.0,
                 horizon: int = 1, ridge: float = 1e-3, sample_spacing: float = 1.0):
        if not isinstance(window, (int, np.integer)) or window < 2:
            raise ValueError("window must be an integer >= 2")
        if not isinstance(order, (int, np.integer)) or not 1 <= order <= window:
            raise ValueError("order must be between 1 and window")
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not np.isfinite(rate) or rate <= 0 or not np.isfinite(sample_spacing) or sample_spacing <= 0:
            raise ValueError("rate and sample_spacing must be finite and positive")
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        self.window, self.order, self.rate = int(window), int(order), float(rate)
        self.horizon, self.ridge, self.sample_spacing = int(horizon), float(ridge), float(sample_spacing)
        lags = np.arange(window) * sample_spacing
        basis = np.column_stack([np.sqrt(2*rate) * np.exp(-rate*lags) * special.eval_laguerre(k, 2*rate*lags)
                                 for k in range(order)])
        self.encoder_ = np.linalg.pinv(basis)

    def _examples(self, values: np.ndarray):
        y = np.asarray(values, dtype=float)
        if y.ndim != 1 or len(y) < self.window + self.horizon or not np.all(np.isfinite(y)):
            raise ValueError("values must be a finite one-dimensional series with enough samples")
        windows = np.lib.stride_tricks.sliding_window_view(y, self.window)
        count = len(y) - self.window - self.horizon + 1
        design = windows[:count, ::-1] @ self.encoder_.T
        target_indices = np.arange(self.window - 1 + self.horizon, len(y))
        return design, y[target_indices], target_indices

    def fit(self, values: np.ndarray, train_end: int | None = None) -> "LaguerreForecaster":
        design, targets, indices = self._examples(values)
        if train_end is None:
            train_end = len(values)
        if not isinstance(train_end, (int, np.integer)) or train_end < 1 or train_end > len(values):
            raise ValueError("train_end must be an index within the series")
        selected = indices < train_end
        if not np.any(selected):
            raise ValueError("train_end leaves no training windows")
        self.intercept_, self.coef_ = _ridge_readout(design[selected], targets[selected], self.ridge)
        self.train_end_ = int(train_end)
        return self

    def predict_from_window(self, history: np.ndarray) -> float:
        """Predict horizon samples after the final value in a history window."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before predict_from_window")
        window = np.asarray(history, dtype=float)
        if window.shape != (self.window,) or not np.all(np.isfinite(window)):
            raise ValueError("history must be a finite vector of length window")
        features = window[::-1] @ self.encoder_.T
        return float(self.intercept_ + features @ self.coef_)

    def predict_series(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return target indices and corresponding history-only predictions."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before predict_series")
        design, _, indices = self._examples(values)
        return indices, self.intercept_ + design @ self.coef_
