"""Reproducible, held-out comparisons on three real datasets.

Install with ``python -m pip install -e '.[benchmark]'`` and run this file.
The sunspots CSV is bundled in examples/data; other datasets are bundled with
scikit-learn and NetworkX. No dataset is downloaded when this script runs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np
import scipy
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import KFold, train_test_split

from orthopoly_applications import (
    ChebyshevGraphClassifier,
    LaguerreForecaster,
    PolynomialFamily,
    PolynomialRegressor,
)


HERE = Path(__file__).resolve().parent


def mae(actual, predicted) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def rmse(actual, predicted) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def regression_benchmark() -> dict:
    dataset = load_diabetes()
    # The library's regressor accepts one scalar feature, so every method gets BMI only.
    x = dataset.data[:, dataset.feature_names.index("bmi")]
    y = dataset.target
    folds = KFold(n_splits=5, shuffle=True, random_state=2026)
    scores = {name: [] for name in ("orthopoly", "linear", "train_mean")}
    chosen = []
    for outer_train, test in folds.split(x):
        fit, validation = train_test_split(outer_train, test_size=0.25, random_state=2026)
        trials = []
        for degree in (2, 3, 4, 5):
            for ridge in (0.0, 0.001, 0.1, 1.0):
                model = PolynomialRegressor(PolynomialFamily("chebyshev"), degree, ridge)
                model.fit(x[fit], y[fit])
                trials.append((mae(y[validation], model.predict(x[validation])), degree, ridge))
        _, degree, ridge = min(trials)
        chosen.append({"degree": degree, "ridge": ridge})
        model = PolynomialRegressor(PolynomialFamily("chebyshev"), degree, ridge).fit(x[outer_train], y[outer_train])
        linear = LinearRegression().fit(x[outer_train, None], y[outer_train])
        predictions = {
            "orthopoly": model.predict(x[test]),
            "linear": linear.predict(x[test, None]),
            "train_mean": np.full(len(test), y[outer_train].mean()),
        }
        for name, values in predictions.items():
            scores[name].append(mae(y[test], values))
    return {
        "dataset": "scikit-learn diabetes",
        "task": "Predict disease progression from BMI only",
        "samples": len(y),
        "evaluation": "5 outer folds; 25% of each outer training fold for model selection",
        "metric": "MAE; lower is better",
        "mean_mae": {name: round(float(np.mean(values)), 3) for name, values in scores.items()},
        "fold_mae": {name: [round(v, 3) for v in values] for name, values in scores.items()},
        "selected_settings": chosen,
    }


def graph_benchmark() -> dict:
    graph = nx.karate_club_graph()
    nodes = sorted(graph.nodes)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight=None)
    labels = np.array([int(graph.nodes[node]["club"] == "Officer") for node in nodes])
    # Structural features are derived from the graph and contain no club labels.
    degrees = np.array([graph.degree[node] for node in nodes], dtype=float)
    clustering = np.array([nx.clustering(graph, node) for node in nodes])
    triangles = np.array([nx.triangles(graph, node) for node in nodes], dtype=float)
    features = np.column_stack([degrees, clustering, triangles])
    class_zero = np.flatnonzero(labels == 0)
    class_one = np.flatnonzero(labels == 1)
    scores = {name: [] for name in ("orthopoly", "harmonic_label_propagation", "feature_logistic", "majority")}
    rng = np.random.default_rng(2026)
    for _ in range(20):
        train = np.r_[rng.choice(class_zero, 4, replace=False), rng.choice(class_one, 4, replace=False)]
        mask = np.ones(len(nodes), dtype=bool)
        mask[train] = False
        model = ChebyshevGraphClassifier(order=2, ridge=0.001).fit(adjacency, features, labels, train)
        logistic = LogisticRegression(max_iter=1000).fit(features[train], labels[train])
        majority = np.full(mask.sum(), int(np.mean(labels[train]) > 0.5))
        laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
        unlabeled = np.flatnonzero(mask)
        probabilities = np.linalg.solve(
            laplacian[np.ix_(unlabeled, unlabeled)],
            adjacency[np.ix_(unlabeled, train)] @ labels[train],
        )
        predictions = {
            "orthopoly": model.predict(adjacency, features)[mask],
            "harmonic_label_propagation": (probabilities >= 0.5).astype(int),
            "feature_logistic": logistic.predict(features[mask]),
            "majority": majority,
        }
        for name, values in predictions.items():
            scores[name].append(float(np.mean(values == labels[mask])))
    return {
        "dataset": "Zachary's Karate Club (NetworkX)",
        "task": "Predict recorded club faction from network structure",
        "nodes": len(nodes),
        "edges": graph.number_of_edges(),
        "evaluation": "20 seeded splits; 4 labeled members per faction, 26 held-out members per split",
        "features": ["degree", "clustering coefficient", "triangle count"],
        "metric": "Accuracy; higher is better",
        "mean_accuracy": {name: round(float(np.mean(values)), 3) for name, values in scores.items()},
        "split_accuracy": {name: [round(v, 3) for v in values] for name, values in scores.items()},
    }


def _sunspots() -> tuple[np.ndarray, np.ndarray]:
    with (HERE / "data" / "sunspots.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    years = np.array([int(row["YEAR"]) for row in rows])
    values = np.array([float(row["SUNACTIVITY"]) for row in rows])
    if len(values) != 309 or years[0] != 1700 or years[-1] != 2008:
        raise ValueError("The bundled yearly sunspots dataset is incomplete")
    return years, values


def _autoregressive_rows(values: np.ndarray, window: int):
    windows = np.lib.stride_tricks.sliding_window_view(values, window)[:-1]
    indices = np.arange(window, len(values))
    return windows, indices


def forecast_benchmark() -> dict:
    years, values = _sunspots()
    train_end, validation_end = 220, 260  # target years 1920 and 1960
    laguerre_trials = []
    for window in (8, 12, 16, 24):
        for order in (3, 5, 8):
            for rate in (0.1, 0.2, 0.4, 0.8):
                for ridge in (0.01, 1.0, 100.0):
                    model = LaguerreForecaster(window, order, rate=rate, ridge=ridge).fit(values, train_end)
                    indices, predicted = model.predict_series(values)
                    validation = (indices >= train_end) & (indices < validation_end)
                    laguerre_trials.append((rmse(values[indices[validation]], predicted[validation]),
                                             window, order, rate, ridge))
    _, window, order, rate, ridge = min(laguerre_trials)
    model = LaguerreForecaster(window, order, rate=rate, ridge=ridge).fit(values, validation_end)
    indices, predicted = model.predict_series(values)
    test = indices >= validation_end
    actual = values[indices[test]]
    laguerre_prediction = predicted[test]

    ar_trials = []
    for ar_window in (4, 8, 12, 16, 24):
        windows, targets = _autoregressive_rows(values, ar_window)
        fit = targets < train_end
        validation = (targets >= train_end) & (targets < validation_end)
        for alpha in (0.0, 1.0, 100.0, 1000.0):
            baseline = Ridge(alpha=alpha).fit(windows[fit], values[targets[fit]])
            ar_trials.append((rmse(values[targets[validation]], baseline.predict(windows[validation])),
                              ar_window, alpha))
    _, ar_window, alpha = min(ar_trials)
    windows, targets = _autoregressive_rows(values, ar_window)
    baseline = Ridge(alpha=alpha).fit(windows[targets < validation_end], values[targets[targets < validation_end]])
    ar_prediction = baseline.predict(windows[targets >= validation_end])
    if not np.array_equal(targets[targets >= validation_end], indices[test]):
        raise AssertionError("Forecast methods must score the same target years")
    persistence_prediction = values[indices[test] - 1]
    predictions = {
        "orthopoly": laguerre_prediction,
        "autoregressive_ridge": ar_prediction,
        "persistence": persistence_prediction,
    }
    return {
        "dataset": "Yearly sunspot counts, 1700–2008 (statsmodels / NGDC)",
        "task": "One-year-ahead forecast using observed history",
        "train_years": [1700, 1919],
        "validation_years": [1920, 1959],
        "test_years": [1960, 2008],
        "test_count": len(actual),
        "metric": "RMSE and MAE; lower is better",
        "test_rmse": {name: round(rmse(actual, p), 3) for name, p in predictions.items()},
        "test_mae": {name: round(mae(actual, p), 3) for name, p in predictions.items()},
        "selected_settings": {"orthopoly": {"window": window, "order": order, "rate": rate, "ridge": ridge},
                              "autoregressive_ridge": {"window": ar_window, "ridge": alpha}},
        "test_years_all": years[indices[test]].tolist(),
        "test_actual": np.round(actual, 2).tolist(),
        "test_predictions": {name: np.round(p, 2).tolist() for name, p in predictions.items()},
    }


def main() -> None:
    result = {
        "description": "Real-data benchmark results; not synthetic demonstration data",
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit_learn": sklearn.__version__, "networkx": nx.__version__},
        "regression": regression_benchmark(),
        "graph": graph_benchmark(),
        "forecast": forecast_benchmark(),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
