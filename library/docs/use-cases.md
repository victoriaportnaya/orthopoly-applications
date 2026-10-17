# Use cases

These examples show how to match a data problem to a working method in Orthopoly Applications. The included runnable demonstrations use deterministic synthetic data. They demonstrate workflows; for measured results and baseline comparisons on real datasets, see [real-data benchmarks](real-data-benchmarks.md).

## Example outputs

Run `python3 examples/use_case_outputs.py` after installing the package. The script prints all six results as JSON; the checked-in `examples/use-case-outputs.json` is the generated snapshot. Values below are rounded for display.

| Case | Output from deterministic sample data |
| --- | --- |
| Measured-response regression | Held-out MAE **0.129** on 16 noisy measurements; for `x=-1.873`, observed `-4.641`, predicted `-4.475`. |
| Partly labeled network | **52/52** held-out nodes labeled correctly on a synthetic two-cluster graph; node 10 → class 0, node 45 → class 1. |
| Future reading | Held-out RMSE **0.061** on 80 later targets; at index 220, observed `0.063`, forecast `0.184`. |
| Fixed graph filter | Input `[0, 0, 1, 0, 0]` → output `[0, 0.25, 0.5, 0.25, 0]`. |
| Weighted projection | Chebyshev coefficients of `x²` at degree 2: `[0.5, 0, 0.5]`. |
| Weyl-orbit sums | For A1 weight 1 at `x=0.25`, `C(x)=0` and `S(x)=2i`. |

The classification graph deliberately separates two clusters, so its perfect score is a property of this toy example. These outputs do not establish performance on another dataset.

| Data | Goal | Method |
| --- | --- | --- |
| Paired scalar inputs and responses | Predict a continuous response | `PolynomialRegressor` |
| One graph with some labeled nodes | Classify other nodes on that graph | `ChebyshevGraphClassifier` |
| One regularly sampled scalar signal | Predict a future sample | `LaguerreForecaster` |
| Node values on an undirected graph | Apply a chosen spectral transform | `chebyshev_graph_filter` |
| A vectorized callable function | Weighted basis approximation | `project_function` |
| Root-system weights and evaluation points | Explore orbit symmetry | `WeylOrbit` |

## 1. Calibrate measured responses

With paired one-dimensional measurements `(x, y)`, fit a low-degree `PolynomialRegressor` and predict at new inputs. A possible setting is an instrument response curve. Select degree and ridge strength on a validation split; score held-out measurements within the training range. Chebyshev and Jacobi inputs are scaled to that range, and predictions beyond it are extrapolations. See [the regression example](training.md).

## 2. Label nodes on a partly known network

Provide a symmetric nonnegative adjacency matrix, node features, known labels, and a training mask. `ChebyshevGraphClassifier` learns from labeled nodes and predicts labels or class probabilities for nodes on the *same* graph. Hold out some known labels for scoring. The graph and unlabeled node features are available during fitting, so this is transductive node classification. It is one polynomial layer with a linear softmax readout, not a general deep GNN. See [the graph example](training.md).

## 3. Forecast the next reading

For a regularly sampled scalar series, `LaguerreForecaster` encodes recent history windows and predicts a future sample at the selected `horizon`. Split chronologically, set `train_end` before the test targets, and tune on an earlier validation interval. This is a sampled Laguerre encoder with a ridge readout, not a recurrent network. See [the forecasting example](training.md).

## 4. Transform values across graph connections

Use `normalized_laplacian(adjacency)` and `chebyshev_graph_filter(laplacian, features, coefficients, lambda_max=2)` for a fixed spectral transform on node values. The coefficient array determines the response; the utility does not learn coefficients from labels. Inspect the output on representative graph signals and compare with the original values. See [the graph utility API](api.md).

## 5. Approximate a known function

`project_function` uses weighted Gaussian quadrature to compute coefficients for a vectorized callable in a supported polynomial family. Reconstruct with `PolynomialFamily.evaluate_series`, then compare against the original function on a grid. This is numerical projection of a callable; use `PolynomialRegressor` instead for noisy observed pairs. See [basis methods](api.md) and [weight conventions](mathematics.md).

## 6. Explore Weyl-orbit symmetry

For A1, A2, B2, or G2, construct a `WeylOrbit` from a dominant integral weight and evaluate C or S exponential sums at dual-coordinate points. The complex outputs are useful for examining orbit and reflection behavior; this is not a trained classifier. Verify the coordinate convention and simple A1 identities. See [orbit methods](api.md) and [mathematical conventions](mathematics.md).
