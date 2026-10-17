# API reference

All objects below are exported from `orthopoly_applications`.

## Trainable models

### `PolynomialRegressor(family, degree, ridge=0.0)`

`fit(x, y) -> self` learns a one-dimensional polynomial response. `predict(x) -> ndarray` evaluates it. `family` is a `PolynomialFamily`. After fitting, `coef_` stores the basis coefficients; for Chebyshev and Jacobi, `domain_` stores the training range.

### `ChebyshevGraphClassifier(order=2, ridge=1e-3, max_iter=300)`

`fit(adjacency, features, labels, train_mask=None) -> self` learns from the selected nodes, or all nodes if `train_mask` is omitted. `predict(adjacency, features) -> ndarray` returns class labels. `predict_proba(...) -> ndarray` returns class probabilities in `classes_` order. Learned attributes include `weights_`, `classes_`, `loss_`, and `n_iter_`. The graph can be dense or SciPy sparse; prediction features must have the training channel count.

### `LaguerreForecaster(window, order, rate=1.0, horizon=1, ridge=1e-3, sample_spacing=1.0)`

`fit(values, train_end=None) -> self` trains on targets before `train_end` (all targets if omitted). `predict_from_window(history) -> float` forecasts from exactly `window` samples. `predict_series(values) -> (target_indices, predictions)` produces history-only predictions for all complete windows. Learned attributes include `coef_`, `intercept_`, and `train_end_`.

## Polynomial and signal utilities

### `PolynomialFamily(kind, alpha=0.0, beta=0.0)`

Supported `kind` values: `"jacobi"`, `"hermite"`, `"hermite_probabilists"`, `"chebyshev"`, and `"laguerre"`. Jacobi parameters require `alpha, beta > -1`; other kinds do not accept nonzero parameters. Methods:

- `evaluate(degree, x)`: standard polynomial value.
- `evaluate_series(coefficients, x)`: sum of coefficients times basis polynomials.
- `quadrature(order)`: Gaussian integration nodes and weights for the family's weight.
- `norm_squared(degree)`: squared norm under that weight.

`project_function(function, family, degree, quadrature_order=None)` returns weighted expansion coefficients for degrees `0..degree`. The callable must accept an array and return a same-shape array. Quadrature is exact for sufficiently low-degree polynomial integrands and approximate otherwise.

`normalized_laplacian(adjacency)` returns `I - D^(-1/2) A D^(-1/2)` with isolated-node diagonal zero. `chebyshev_graph_filter(laplacian, features, coefficients, lambda_max=None)` applies a fixed polynomial spectral filter. Pass `lambda_max=2` for a normalized Laplacian to avoid an eigenvalue calculation on a large graph.

`chebyshev_smooth(values, degree)` least-squares fits a degree-`degree` Chebyshev trend to uniformly spaced samples and returns the fitted samples.

`LaguerreExpansion.fit(times, values, order, rate=1.0)` fits a finite sampled signal to Laguerre functions. The resulting object has `coefficients` and `reconstruct(times)`.

## Orbit functions

`cartan_matrix(type_name)` returns a supported Cartan matrix for `A1`, `A2`, `B2`, or `G2`. `WeylOrbit(cartan, weight)` takes a dominant integral weight in fundamental-weight coordinates. `orbit()` returns distinct weights and reflection-parity signs; `evaluate(points, kind="C")` evaluates the complex C or S exponential sum. Points have final dimension equal to the rank and use coordinates dual to the chosen weights. S is zero for weights on a reflection wall.
