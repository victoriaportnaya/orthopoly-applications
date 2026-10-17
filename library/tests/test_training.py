import unittest

import numpy as np

from orthopoly_applications import (
    PolynomialFamily, PolynomialRegressor,
    ChebyshevGraphClassifier, LaguerreForecaster,
)


class TrainingTests(unittest.TestCase):
    def test_polynomial_regression_generalizes_within_range(self):
        x = np.linspace(-2, 3, 50)
        y = 1 + 2*x - 0.5*x*x
        model = PolynomialRegressor(PolynomialFamily("chebyshev"), degree=2).fit(x, y)
        test_x = np.linspace(-1.9, 2.9, 21)
        np.testing.assert_allclose(model.predict(test_x), 1 + 2*test_x - 0.5*test_x**2, atol=1e-12)
        with self.assertRaises(ValueError):
            PolynomialRegressor(PolynomialFamily("laguerre"), 2).fit(x, y)

    def test_graph_classifier_uses_only_training_labels(self):
        rng = np.random.default_rng(7)
        n = 60
        adjacency = np.zeros((n, n))
        adjacency[:30, :30] = adjacency[30:, 30:] = 1
        np.fill_diagonal(adjacency, 0)
        adjacency[29, 30] = adjacency[30, 29] = 1
        labels = np.r_[np.zeros(30, dtype=int), np.ones(30, dtype=int)]
        features = np.r_[np.full(30, -1.), np.full(30, 1.)] + rng.normal(0, 1.2, n)
        train = np.zeros(n, dtype=bool)
        train[[0, 1, 2, 3, 30, 31, 32, 33]] = True
        model = ChebyshevGraphClassifier(order=2).fit(adjacency, features, labels, train)
        prediction = model.predict(adjacency, features)
        self.assertGreater(np.mean(prediction[~train] == labels[~train]), 0.9)
        np.testing.assert_allclose(model.predict_proba(adjacency, features).sum(axis=1), 1)
        changed_holdout = labels.copy()
        changed_holdout[~train] = 1 - changed_holdout[~train]
        model2 = ChebyshevGraphClassifier(order=2).fit(adjacency, features, changed_holdout, train)
        np.testing.assert_array_equal(prediction, model2.predict(adjacency, features))

    def test_laguerre_forecaster_holdout_and_no_leakage(self):
        t = np.arange(300)
        signal = np.sin(0.2*t) + 0.4*np.sin(0.07*t)
        params = dict(window=20, order=10, rate=0.25, ridge=1e-5)
        model = LaguerreForecaster(**params).fit(signal, train_end=220)
        prefix_model = LaguerreForecaster(**params).fit(signal[:220])
        np.testing.assert_allclose(model.coef_, prefix_model.coef_, atol=1e-10)
        indices, predictions = model.predict_series(signal)
        holdout = indices >= 220
        error = np.sqrt(np.mean((predictions[holdout] - signal[indices[holdout]])**2))
        persistence_error = np.sqrt(np.mean((signal[indices[holdout]-1] - signal[indices[holdout]])**2))
        self.assertLess(error, persistence_error / 10)
        self.assertAlmostEqual(model.predict_from_window(signal[200-20:200]),
                               predictions[np.where(indices == 200)[0][0]], places=10)

    def test_unfitted_models_reject_prediction(self):
        with self.assertRaises(RuntimeError):
            PolynomialRegressor(PolynomialFamily("chebyshev"), 2).predict([0])
        with self.assertRaises(RuntimeError):
            ChebyshevGraphClassifier().predict(np.eye(2), [0, 1])
        with self.assertRaises(RuntimeError):
            LaguerreForecaster(5, 2).predict_from_window(np.zeros(5))


if __name__ == "__main__":
    unittest.main()
