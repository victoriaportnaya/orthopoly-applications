"""Numerical applications of classical orthogonal polynomials and orbit sums."""

from .families import PolynomialFamily, project_function
from .graph import chebyshev_graph_filter, normalized_laplacian
from .signals import LaguerreExpansion, chebyshev_smooth
from .orbits import WeylOrbit, cartan_matrix
from .training import PolynomialRegressor, ChebyshevGraphClassifier, LaguerreForecaster

__all__ = [
    "PolynomialFamily",
    "project_function",
    "chebyshev_graph_filter",
    "normalized_laplacian",
    "LaguerreExpansion",
    "chebyshev_smooth",
    "WeylOrbit",
    "cartan_matrix",
    "PolynomialRegressor",
    "ChebyshevGraphClassifier",
    "LaguerreForecaster",
]
