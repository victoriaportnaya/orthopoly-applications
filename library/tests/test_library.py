import unittest

import numpy as np
from scipy import sparse

from orthopoly_applications import (
    PolynomialFamily, project_function, normalized_laplacian,
    chebyshev_graph_filter, chebyshev_smooth, LaguerreExpansion,
    WeylOrbit, cartan_matrix,
)


class ClassicalFamiliesTests(unittest.TestCase):
    def test_reference_values_and_norms(self):
        x = np.array([-0.5, 0., 0.5])
        np.testing.assert_allclose(PolynomialFamily("chebyshev").evaluate(2, x), 2*x*x - 1)
        np.testing.assert_allclose(PolynomialFamily("hermite").evaluate(2, x), 4*x*x - 2)
        np.testing.assert_allclose(PolynomialFamily("hermite_probabilists").evaluate(2, x), x*x - 1)
        np.testing.assert_allclose(PolynomialFamily("laguerre").evaluate(2, x), 1 - 2*x + x*x/2)
        # For alpha=beta=-1/2, Jacobi's weight is the Chebyshev weight.
        self.assertAlmostEqual(PolynomialFamily("jacobi", -0.5, -0.5).norm_squared(0), np.pi)

    def test_projection_recovers_polynomial(self):
        for family in [PolynomialFamily("jacobi", 0.3, 0.7), PolynomialFamily("hermite"),
                       PolynomialFamily("hermite_probabilists"), PolynomialFamily("chebyshev"),
                       PolynomialFamily("laguerre")]:
            coeff = project_function(lambda x: 1 + 2*x + x*x, family, 2)
            grid = np.array([0., 0.25, 0.75])
            reconstructed = family.evaluate_series(coeff, grid)
            np.testing.assert_allclose(reconstructed, 1 + 2*grid + grid*grid, atol=1e-11)

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            PolynomialFamily("jacobi", -1, 0)
        with self.assertRaises(ValueError):
            project_function(lambda x: x, PolynomialFamily("chebyshev"), 3, quadrature_order=3)


class ApplicationTests(unittest.TestCase):
    def test_graph_filter_matches_matrix_polynomial(self):
        adjacency = np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
        L = normalized_laplacian(adjacency)
        X = np.array([[1., 0.], [0., 1.], [1., 1.]])
        c = [0.3, -0.4, 0.7]
        S = L - np.eye(3)  # lambda_max = 2
        expected = c[0]*X + c[1]*(S @ X) + c[2]*((2*S@S - np.eye(3)) @ X)
        np.testing.assert_allclose(chebyshev_graph_filter(L, X, c, lambda_max=2), expected)
        np.testing.assert_allclose(chebyshev_graph_filter(sparse.csr_matrix(L), X, c, lambda_max=2), expected)

    def test_chebyshev_smoothing_preserves_low_degree_signal(self):
        x = np.linspace(-1, 1, 21)
        expected = 1 + x + 2*x*x
        np.testing.assert_allclose(chebyshev_smooth(expected, 2), expected, atol=1e-12)

    def test_laguerre_reconstructs_basis_signal(self):
        t = np.linspace(0, 6, 100)
        basis0 = np.sqrt(2) * np.exp(-t)
        basis1 = basis0 * (1 - 2*t)
        signal = np.column_stack([basis0 + 0.5*basis1, 2*basis0 - basis1])
        memory = LaguerreExpansion.fit(t, signal, order=2, rate=1.)
        np.testing.assert_allclose(memory.coefficients, [[1., 2.], [0.5, -1.]], atol=1e-12)
        np.testing.assert_allclose(memory.reconstruct(t), signal, atol=1e-12)

    def test_orbits_and_signs(self):
        a1 = WeylOrbit(cartan_matrix("A1"), (1,))
        x = np.array([[0.125], [0.25]])
        np.testing.assert_allclose(a1.evaluate(x, "C"), 2*np.cos(2*np.pi*x[:, 0]))
        np.testing.assert_allclose(a1.evaluate(x, "S"), 2j*np.sin(2*np.pi*x[:, 0]))
        self.assertEqual(len(WeylOrbit(cartan_matrix("A2"), (1, 0)).orbit()[0]), 3)
        self.assertEqual(len(WeylOrbit(cartan_matrix("A2"), (1, 1)).orbit()[0]), 6)
        np.testing.assert_allclose(WeylOrbit(cartan_matrix("A2"), (1, 0)).evaluate(np.array([[0.1, 0.2]]), "S"), 0)


if __name__ == "__main__":
    unittest.main()
