# Real-data benchmarks

Three trainable methods were tested on publicly documented, non-synthetic datasets. The benchmark script, bundled sunspots CSV, and machine-readable results are in `examples/`. The numbers below were produced with NumPy 2.4.1, SciPy 1.17.0, scikit-learn 1.8.0, and NetworkX 3.5. They describe these datasets and splits, not general performance guarantees.

| Task and held-out metric | Orthopoly | Comparison methods | Reading |
| --- | ---: | ---: | --- |
| Diabetes, BMI-only regression; 5-fold mean absolute error (lower is better) | **52.236** | Linear regression 51.955; training mean 65.808 | The polynomial model improves on predicting the mean but is slightly worse than a straight line. |
| Karate Club, node classification; mean accuracy over 20 labeled-node splits (higher is better) | **64.8%** | Harmonic graph label propagation 90.8%; feature-only logistic 57.1%; majority 50.0% | Graph smoothing through label propagation is much stronger on this small network. |
| Sunspots, 1960–2008 one-year-ahead forecast; root mean squared error (lower is better) | **16.664** | Autoregressive ridge 18.481; previous-year persistence 30.431 | The Laguerre readout wins this single chronological test. |

## Protocol

**Regression.** The [scikit-learn diabetes dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) contains 442 observations and ten features. `PolynomialRegressor` accepts one scalar input, so every method receives only the supplied BMI feature. Five seeded outer folds evaluate unseen samples. Inside each outer training fold, 25% is reserved to choose polynomial degree 2–5 and ridge strength. The selected model is refit on the full outer training fold, then scored on the untouched test fold. The linear and training-mean baselines see the same outer training rows and feature.

**Graph.** [Zachary's Karate Club graph](https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html) has 34 people and their recorded club faction. Each of 20 seeded splits reveals four labels per faction; the other 26 are scored. All methods receive the same unlabeled graph structure. Structural node features are degree, clustering coefficient, and triangle count; no label is used to construct them. `ChebyshevGraphClassifier` uses order 2 and ridge 0.001. Baselines are a logistic classifier on the same three features, harmonic label propagation on the graph with only the eight training labels, and the training majority class. The graph classifier is transductive: graph edges and held-out node features are visible during fitting.

**Forecasting.** The [statsmodels yearly sunspots series](https://www.statsmodels.org/v0.13.5/datasets/generated/sunspots.html) contains 309 yearly counts from 1700 to 2008 and is bundled as a public-domain CSV. Models train on targets through 1919, choose settings from targets in 1920–1959, refit using targets through 1959, and score 1960–2008. Each one-year-ahead prediction uses observed prior years, including the previous year in the test period; this is rolling one-step evaluation, not a 49-year recursive forecast. Both the Laguerre model and an autoregressive ridge baseline select their window and regularization using only the validation period. Persistence predicts the previous year's count.

## Reproduce

From the package directory:

```bash
python3 -m pip install -e .
python3 -m pip install -r examples/benchmark-requirements.txt
python3 examples/benchmark_real_data.py > examples/real-data-results.json
```

The generated JSON includes fold or split metrics, selected settings, and all held-out sunspot forecasts. The checked-in file is the generated snapshot. The script does not download data: diabetes and Karate Club are bundled with their respective Python packages, and the sunspot CSV is bundled here.

## Limits

These are small, deliberately interpretable comparisons. Diabetes uses just BMI because the current regressor is one-dimensional. Karate Club is one 34-node network, so repeating label splits does not create independent networks. The sunspot result is one historical series and one time split. More datasets, uncertainty estimates, and resource measurements are needed before making broader claims.
