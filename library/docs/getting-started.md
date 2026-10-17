# Getting started

## Install

Use Python 3.10 or newer. From the package directory:

```bash
python3 -m pip install -e .
```

The package requires NumPy and SciPy. A local editable installation lets you change the source and run the examples without reinstalling. To install from a wheel, use `python3 -m pip install path/to/orthopoly_applications-0.1.1-py3-none-any.whl`.

## Run an example

```bash
python3 examples/train_applications.py
```

The script trains three models and prints a regression error, held-out node accuracy, and held-out forecast error. It uses synthetic data with a fixed random seed, so it is a workflow demonstration rather than a benchmark.

## Array shapes

| Task | Inputs | Targets | Prediction |
| --- | --- | --- | --- |
| Polynomial regression | `x`: `(samples,)` | `y`: `(samples,)` | `(new_samples,)` |
| Graph classification | `adjacency`: `(nodes, nodes)`; `features`: `(nodes,)` or `(nodes, channels)` | `labels`: `(nodes,)`; `train_mask`: `(nodes,)` Boolean or node indices | `(nodes,)` labels or `(nodes, classes)` probabilities |
| Laguerre forecast | `signal`: `(time_steps,)` | Future values are derived from `signal` using `horizon` | `(target_indices, predictions)` |

For graph learning, the adjacency matrix must be symmetric with finite, nonnegative weights. A SciPy sparse matrix is accepted. The classifier sees the entire graph and all features during training, but uses labels only at the selected training nodes. This is transductive node classification.

For forecasting, `train_end` is an exclusive index for training **targets**. A forecast at index `t` uses only a window ending at `t - horizon`.

## Next steps

- Read [Train models](training.md) for train/test examples and data-split guidance.
- Read the [API reference](api.md) for all supported families and utility functions.
