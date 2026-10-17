"""Signal smoothing and finite-window Laguerre memory features."""

from dataclasses import dataclass

import numpy as np
from scipy import special


def chebyshev_smooth(values: np.ndarray, degree: int) -> np.ndarray:
    """Fit a low-degree Chebyshev series to uniformly spaced samples.

    Returns the fitted values at the same sample locations. This is a
    least-squares trend/smoothing operation, not a frequency-domain filter.
    """
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or len(y) < 2 or not np.all(np.isfinite(y)):
        raise ValueError("values must be a finite one-dimensional signal of length >= 2")
    if not isinstance(degree, (int, np.integer)) or degree < 0 or degree >= len(y):
        raise ValueError("degree must be between 0 and len(values)-1")
    x = np.linspace(-1, 1, len(y))
    coeff = np.polynomial.chebyshev.chebfit(x, y, degree)
    return np.polynomial.chebyshev.chebval(x, coeff)


@dataclass(frozen=True)
class LaguerreExpansion:
    """A finite Laguerre-function expansion of a sampled causal signal.

    Basis functions are phi_n(t) = sqrt(2a) exp(-at) L_n(2at), t >= 0.
    They are orthonormal on [0, infinity). The supplied time grid covers a
    finite interval, so coefficients and reconstructions are approximations.
    Each column of a multichannel signal is expanded independently.
    """

    coefficients: np.ndarray
    rate: float

    @classmethod
    def fit(cls, times: np.ndarray, values: np.ndarray, order: int, rate: float = 1.0) -> "LaguerreExpansion":
        t = np.asarray(times, dtype=float)
        y = np.asarray(values, dtype=float)
        if t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)) or t[0] < 0 or np.any(np.diff(t) <= 0):
            raise ValueError("times must be finite, nonnegative, and strictly increasing")
        if y.ndim not in (1, 2) or y.shape[0] != len(t) or not np.all(np.isfinite(y)):
            raise ValueError("values must have a finite sample at every time")
        if not isinstance(order, (int, np.integer)) or order < 1 or order > len(t):
            raise ValueError("order must be between 1 and the sample count")
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("rate must be finite and positive")
        basis = _laguerre_basis(t, order, rate)
        # Solve a sampled least-squares problem. It recovers a signal in the
        # finite basis exactly and avoids finite-window projection bias.
        coefficients, *_ = np.linalg.lstsq(basis, y, rcond=None)
        return cls(np.asarray(coefficients), float(rate))

    def reconstruct(self, times: np.ndarray | float) -> np.ndarray:
        """Evaluate the fitted signal at nonnegative times (including delays)."""
        t = np.asarray(times, dtype=float)
        if not np.all(np.isfinite(t)) or np.any(t < 0):
            raise ValueError("times must be finite and nonnegative")
        coeff = np.asarray(self.coefficients)
        if coeff.ndim not in (1, 2) or coeff.shape[0] == 0:
            raise ValueError("coefficients must have shape (order,) or (order, channels)")
        flat = _laguerre_basis(t.ravel(), coeff.shape[0], self.rate) @ coeff
        return flat.reshape(t.shape + coeff.shape[1:])


def _laguerre_basis(times: np.ndarray, order: int, rate: float) -> np.ndarray:
    z = 2 * rate * times
    envelope = np.sqrt(2 * rate) * np.exp(-rate * times)
    return np.column_stack([envelope * special.eval_laguerre(n, z) for n in range(order)])
