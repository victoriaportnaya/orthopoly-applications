"""Train and evaluate all three supervised application models."""

import numpy as np

from orthopoly_applications import (
    PolynomialFamily, PolynomialRegressor,
    ChebyshevGraphClassifier, LaguerreForecaster,
)


# Polynomial regression: the input interval is learned from training data.
x_train = np.linspace(-2, 3, 50)
y_train = 1 + 2*x_train - 0.5*x_train**2
regressor = PolynomialRegressor(PolynomialFamily("chebyshev"), degree=2).fit(x_train, y_train)
x_test = np.linspace(-1.9, 2.9, 30)
y_test = 1 + 2*x_test - 0.5*x_test**2
print("Polynomial regression MAE:", np.mean(np.abs(regressor.predict(x_test) - y_test)))

# Node classification: train only on eight labeled nodes, score on the rest.
rng = np.random.default_rng(7)
n = 60
adjacency = np.zeros((n, n))
adjacency[:30, :30] = 1
adjacency[30:, 30:] = 1
np.fill_diagonal(adjacency, 0)
adjacency[29, 30] = adjacency[30, 29] = 1
labels = np.r_[np.zeros(30, dtype=int), np.ones(30, dtype=int)]
features = np.r_[np.full(30, -1.), np.full(30, 1.)] + rng.normal(0, 1.2, n)
train_mask = np.zeros(n, dtype=bool)
train_mask[[0, 1, 2, 3, 30, 31, 32, 33]] = True
classifier = ChebyshevGraphClassifier(order=2).fit(adjacency, features, labels, train_mask)
predictions = classifier.predict(adjacency, features)
print("Held-out node accuracy:", np.mean(predictions[~train_mask] == labels[~train_mask]))

# Forecasting: targets at indices >= 220 are never used during training.
t = np.arange(300)
signal = np.sin(0.2*t) + 0.4*np.sin(0.07*t)
forecaster = LaguerreForecaster(window=20, order=10, rate=0.25, ridge=1e-5)
forecaster.fit(signal, train_end=220)
target_indices, forecast = forecaster.predict_series(signal)
held_out = target_indices >= 220
rmse = np.sqrt(np.mean((forecast[held_out] - signal[target_indices[held_out]])**2))
print("Held-out forecast RMSE:", rmse)
