# Train models

For held-out results on real datasets and comparisons with baseline methods, see [real-data benchmarks](real-data-benchmarks.md).

Every trainable model uses `fit` followed by `predict` or `predict_series`. Keep a final test set separate when tuning degree, graph order, Laguerre order, rate, or ridge strength.

## Polynomial regression

```python
import numpy as np
from orthopoly_applications import PolynomialFamily, PolynomialRegressor

x_train = np.linspace(-2, 3, 50)
y_train = 1 + 2*x_train - 0.5*x_train**2
model = PolynomialRegressor(PolynomialFamily("chebyshev"), degree=2, ridge=0.0)
model.fit(x_train, y_train)

x_test = np.array([-1.5, 0.5, 2.5])
predictions = model.predict(x_test)
```

For Chebyshev and Jacobi, `fit` scales the training input range to `[-1, 1]` and reuses that scale during prediction. Values outside the training range are extrapolations. Hermite uses the raw real-valued input. Laguerre requires nonnegative input. The degree-zero term is not ridge-penalized.

## Graph node classification

```python
import numpy as np
from orthopoly_applications import ChebyshevGraphClassifier

adjacency = np.array([
    [0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0],
], dtype=float)
features = np.array([-1.2, -0.8, -1.0, 1.1, 0.9, 1.2])
labels = np.array([0, 0, 0, 1, 1, 1])
train_mask = np.array([True, True, False, True, True, False])

model = ChebyshevGraphClassifier(order=2, ridge=1e-3)
model.fit(adjacency, features, labels, train_mask)
predictions = model.predict(adjacency, features)
held_out_accuracy = np.mean(predictions[~train_mask] == labels[~train_mask])
probabilities = model.predict_proba(adjacency, features)
```

The model forms `T_k(L - I) X` for orders `k = 0..order`, where `L` is the normalized Laplacian and `2` is its spectral upper bound. It learns a softmax classifier from the selected node labels using L-BFGS. The model is a single polynomial graph layer with a linear readout; it is not a deep GNN.

## Time-series forecasting

```python
import numpy as np
from orthopoly_applications import LaguerreForecaster

t = np.arange(300)
signal = np.sin(0.2*t) + 0.4*np.sin(0.07*t)
model = LaguerreForecaster(window=20, order=10, rate=0.25, horizon=1, ridge=1e-5)
model.fit(signal, train_end=220)
target_indices, predictions = model.predict_series(signal)

test = target_indices >= 220
rmse = np.sqrt(np.mean((predictions[test] - signal[target_indices[test]])**2))
next_value = model.predict_from_window(signal[-20:])
```

Each history window is reversed so lag zero is its newest sample, encoded with Laguerre functions, and mapped to a future target by a ridge-regularized readout. `train_end=220` uses only targets before index 220. The `predict_series` output includes both training and later predictions; filter by `target_indices` for held-out evaluation.

If you select hyperparameters using a validation period, reserve a later untouched period for the final test. Compare against a simple persistence forecast (`signal[t-1]`) or another task-specific baseline.
