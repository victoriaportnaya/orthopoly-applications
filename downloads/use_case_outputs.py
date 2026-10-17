"""Generate deterministic, illustrative outputs for all six documented use cases.

Run from the package directory after `python3 -m pip install -e .`.
The data is synthetic; these numbers are examples, not real-world benchmarks.
"""

import json
import numpy as np

from orthopoly_applications import (
    PolynomialFamily, PolynomialRegressor, ChebyshevGraphClassifier,
    LaguerreForecaster, normalized_laplacian, chebyshev_graph_filter,
    project_function, WeylOrbit, cartan_matrix,
)


def numbers(values, places=6):
    rounded = np.round(np.asarray(values, dtype=float), places)
    return np.where(rounded == 0, 0.0, rounded).tolist()


def generate():
    # 1. Regression: fit noisy observations, score held-out observations.
    rng = np.random.default_rng(24)
    x = np.linspace(-2, 3, 80)
    true_response = 1 + 2 * x - 0.5 * x**2
    measured = true_response + rng.normal(0, 0.12, len(x))
    test = np.arange(len(x)) % 5 == 2
    regressor = PolynomialRegressor(PolynomialFamily("chebyshev"), degree=2, ridge=1e-3)
    regressor.fit(x[~test], measured[~test])
    predicted = regressor.predict(x[test])
    regression = {
        "train_samples": int((~test).sum()), "test_samples": int(test.sum()),
        "held_out_mae": round(float(np.mean(np.abs(predicted - measured[test]))), 6),
        "sample_x": numbers(x[test][:4], 3),
        "sample_observed": numbers(measured[test][:4], 3),
        "sample_predicted": numbers(predicted[:4], 3),
    }

    # 2. Transductive node classification: labels only at selected nodes.
    rng = np.random.default_rng(7)
    n = 60
    adjacency = np.zeros((n, n))
    adjacency[:30, :30] = 1
    adjacency[30:, 30:] = 1
    np.fill_diagonal(adjacency, 0)
    adjacency[29, 30] = adjacency[30, 29] = 1
    labels = np.r_[np.zeros(30, dtype=int), np.ones(30, dtype=int)]
    features = np.r_[np.full(30, -1.), np.full(30, 1.)] + rng.normal(0, 1.2, n)
    train = np.zeros(n, dtype=bool)
    train[[0, 1, 2, 3, 30, 31, 32, 33]] = True
    classifier = ChebyshevGraphClassifier(order=2).fit(adjacency, features, labels, train)
    classified = classifier.predict(adjacency, features)
    probabilities = classifier.predict_proba(adjacency, features)
    sample_nodes = [10, 45]
    graph = {
        "labeled_nodes": int(train.sum()), "held_out_nodes": int((~train).sum()),
        "held_out_correct": int(np.sum(classified[~train] == labels[~train])),
        "held_out_accuracy": round(float(np.mean(classified[~train] == labels[~train])), 6),
        "sample_nodes": sample_nodes,
        "sample_predicted_labels": classified[sample_nodes].astype(int).tolist(),
        "sample_class_1_probabilities": numbers(probabilities[sample_nodes, 1], 3),
    }

    # 3. Forecasting: train only on targets before index 220.
    rng = np.random.default_rng(19)
    t = np.arange(300)
    signal = np.sin(0.2 * t) + 0.4 * np.sin(0.07 * t) + rng.normal(0, 0.05, len(t))
    forecaster = LaguerreForecaster(window=20, order=10, rate=0.25, ridge=1e-5)
    forecaster.fit(signal, train_end=220)
    target_indices, forecasts = forecaster.predict_series(signal)
    held_out = target_indices >= 220
    errors = forecasts[held_out] - signal[target_indices[held_out]]
    forecast = {
        "train_end_exclusive": 220, "held_out_targets": int(held_out.sum()),
        "held_out_rmse": round(float(np.sqrt(np.mean(errors**2))), 6),
        "sample_target_indices": target_indices[held_out][:4].astype(int).tolist(),
        "sample_observed": numbers(signal[target_indices[held_out][:4]], 3),
        "sample_predicted": numbers(forecasts[held_out][:4], 3),
    }

    # 4. A fixed low-pass-like transform on a five-node path.
    path = np.zeros((5, 5))
    for i in range(4):
        path[i, i + 1] = path[i + 1, i] = 1
    node_values = np.array([0., 0., 1., 0., 0.])
    filtered = chebyshev_graph_filter(normalized_laplacian(path), node_values,
                                      [0.5, -0.5], lambda_max=2)
    graph_filter = {
        "coefficients": [0.5, -0.5], "input_values": numbers(node_values, 3),
        "output_values": numbers(filtered, 3),
    }

    # 5. Exact degree-two representation of x^2 in a Chebyshev basis.
    family = PolynomialFamily("chebyshev")
    coefficients = project_function(lambda points: points**2, family, degree=2)
    points = np.array([-1., -0.5, 0., 0.5, 1.])
    reconstruction = family.evaluate_series(coefficients, points)
    projection = {
        "function": "x^2", "family": "chebyshev", "degree": 2,
        "coefficients": numbers(coefficients, 6), "sample_points": numbers(points, 3),
        "reconstructed_values": numbers(reconstruction, 3),
        "max_abs_error_on_sample": round(float(np.max(np.abs(reconstruction - points**2))), 12),
    }

    # 6. C and S orbit sums for A1 weight 1.
    orbit = WeylOrbit(cartan_matrix("A1"), (1,))
    orbit_points = np.array([[0.], [0.125], [0.25], [0.5]])
    c_values = orbit.evaluate(orbit_points, kind="C")
    s_values = orbit.evaluate(orbit_points, kind="S")
    symmetry = {
        "system": "A1", "weight": [1], "points": numbers(orbit_points.ravel(), 3),
        "C_real": numbers(np.real(c_values), 3),
        "S_imag": numbers(np.imag(s_values), 3),
    }

    return {
        "note": "Deterministic synthetic examples; rounded for display; not real-world benchmarks.",
        "regression": regression, "node_classification": graph,
        "forecasting": forecast, "graph_filter": graph_filter,
        "projection": projection, "symmetry": symmetry,
    }


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
