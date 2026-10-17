"""Classical polynomial families and Gaussian-quadrature projections."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import special


@dataclass(frozen=True)
class PolynomialFamily:
    """A standard (not orthonormalized) classical polynomial family.

    ``kind`` is ``jacobi``, ``hermite`` (physicists' convention),
    ``hermite_probabilists``, ``chebyshev``, or ``laguerre``.
    Jacobi requires alpha,beta > -1. Other families take no parameters.
    """

    kind: str
    alpha: float = 0.0
    beta: float = 0.0

    def __post_init__(self) -> None:
        valid = {"jacobi", "hermite", "hermite_probabilists", "chebyshev", "laguerre"}
        if self.kind not in valid:
            raise ValueError(f"kind must be one of {sorted(valid)}")
        if self.kind == "jacobi":
            if not np.isfinite(self.alpha) or not np.isfinite(self.beta) or self.alpha <= -1 or self.beta <= -1:
                raise ValueError("Jacobi alpha and beta must be finite and > -1")
        elif self.alpha != 0 or self.beta != 0:
            raise ValueError("alpha and beta are only used for Jacobi polynomials")

    def evaluate(self, degree: int, x: np.ndarray | float) -> np.ndarray:
        """Evaluate the degree ``degree`` polynomial at x."""
        if not isinstance(degree, (int, np.integer)) or degree < 0:
            raise ValueError("degree must be a nonnegative integer")
        x = np.asarray(x, dtype=float)
        if self.kind == "jacobi":
            return special.eval_jacobi(degree, self.alpha, self.beta, x)
        if self.kind == "hermite":
            return special.eval_hermite(degree, x)
        if self.kind == "hermite_probabilists":
            return special.eval_hermitenorm(degree, x)
        if self.kind == "chebyshev":
            return special.eval_chebyt(degree, x)
        return special.eval_laguerre(degree, x)

    def quadrature(self, order: int) -> tuple[np.ndarray, np.ndarray]:
        """Return nodes and weights for the family's weighted integral."""
        if not isinstance(order, (int, np.integer)) or order < 1:
            raise ValueError("order must be a positive integer")
        if self.kind == "jacobi":
            return special.roots_jacobi(order, self.alpha, self.beta)
        if self.kind == "hermite":
            return special.roots_hermite(order)
        if self.kind == "hermite_probabilists":
            return special.roots_hermitenorm(order)
        if self.kind == "chebyshev":
            return special.roots_chebyt(order)
        return special.roots_laguerre(order)

    def evaluate_series(self, coefficients: np.ndarray, x: np.ndarray | float) -> np.ndarray:
        """Evaluate a series with coefficient[n] multiplying degree n."""
        c = np.asarray(coefficients)
        if c.ndim != 1 or len(c) == 0 or not np.all(np.isfinite(c)):
            raise ValueError("coefficients must be a finite nonempty vector")
        result = np.zeros_like(np.asarray(x), dtype=np.result_type(c, float))
        for degree, value in enumerate(c):
            result = result + value * self.evaluate(degree, x)
        return result

    def norm_squared(self, degree: int) -> float:
        """Squared L2 norm under the family's standard weight."""
        if not isinstance(degree, (int, np.integer)) or degree < 0:
            raise ValueError("degree must be a nonnegative integer")
        n = int(degree)
        if self.kind == "jacobi":
            a, b = self.alpha, self.beta
            if n == 0:
                return float(np.exp((a+b+1)*np.log(2) + special.betaln(a+1, b+1)))
            # Gamma-ratio form remains stable for moderately large degrees.
            log_norm = ((a + b + 1) * np.log(2) - np.log(2*n + a + b + 1)
                        + special.gammaln(n+a+1) + special.gammaln(n+b+1)
                        - special.gammaln(n+1) - special.gammaln(n+a+b+1))
            return float(np.exp(log_norm))
        if self.kind == "hermite":
            return float(np.exp(0.5*np.log(np.pi) + n*np.log(2) + special.gammaln(n+1)))
        if self.kind == "hermite_probabilists":
            return float(np.exp(0.5*np.log(2*np.pi) + special.gammaln(n+1)))
        if self.kind == "chebyshev":
            return float(np.pi if n == 0 else np.pi/2)
        return 1.0


def project_function(
    function: Callable[[np.ndarray], np.ndarray],
    family: PolynomialFamily,
    degree: int,
    quadrature_order: int | None = None,
) -> np.ndarray:
    """Project f onto degrees 0..degree in the family's weighted L2 space.

    Gaussian quadrature approximates the inner products. It is exact for
    polynomial f when its order integrates f times each basis polynomial.
    ``function`` must accept a NumPy array and return one value per node.
    """
    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    order = quadrature_order if quadrature_order is not None else max(32, 2*degree + 4)
    if order <= degree:
        raise ValueError("quadrature_order must exceed degree")
    nodes, weights = family.quadrature(order)
    values = np.asarray(function(nodes))
    if values.shape != nodes.shape or not np.all(np.isfinite(values)):
        raise ValueError("function must return finite values with the nodes' shape")
    return np.asarray([
        np.dot(weights, values * family.evaluate(n, nodes)) / family.norm_squared(n)
        for n in range(degree + 1)
    ])
