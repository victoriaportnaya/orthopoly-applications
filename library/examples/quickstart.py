"""Run three small application examples."""

import numpy as np

from orthopoly_applications import (
    PolynomialFamily, project_function, normalized_laplacian,
    chebyshev_graph_filter, LaguerreExpansion, WeylOrbit, cartan_matrix,
)


family = PolynomialFamily("chebyshev")
coeff = project_function(lambda x: 1 + 2*x + x*x, family, 2)
print("Chebyshev coefficients:", np.round(coeff, 6))

adjacency = np.array([[0., 1.], [1., 0.]])
laplacian = normalized_laplacian(adjacency)
print("Graph filter:", chebyshev_graph_filter(laplacian, np.array([1., 0.]), [1., -0.5], lambda_max=2.))

t = np.linspace(0, 5, 100)
memory = LaguerreExpansion.fit(t, np.exp(-t) * (1 + t), order=3)
print("Laguerre values:", np.round(memory.reconstruct(np.array([0., 1., 2.])), 6))

orbit = WeylOrbit(cartan_matrix("A1"), (1,))
print("A1 C orbit:", orbit.evaluate(np.array([[0., 0.25, 0.5]]).T, "C"))
