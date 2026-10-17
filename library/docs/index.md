# Orthopoly Applications documentation

Orthopoly Applications turns four classical orthogonal-polynomial families into practical Python tools. It includes trainable models for one-dimensional regression, graph node classification, and time-series forecasting, plus basis evaluation, graph filtering, signal smoothing, and low-rank Weyl orbit sums.

## Choose a guide

1. [Getting started](getting-started.md): install, run the examples, and understand the expected array shapes.
2. [Train models](training.md): prepare data, fit each estimator, and evaluate held-out results.
3. [Real-data benchmarks](real-data-benchmarks.md): compare each trainable method with baselines on held-out data.
4. [API reference](api.md): signatures, return values, and constraints for each public object.
5. [Mathematical conventions](mathematics.md): weights, basis normalizations, graph scaling, and orbit coordinates.

The package needs Python 3.10+, NumPy, and SciPy. It does not need PyTorch or an internet service to train the included models.

## Quick commands

Run these from the `orthopoly_applications` directory:

```bash
python3 -m pip install -e .
python3 examples/train_applications.py
python3 -m unittest discover -s tests -v
```

The training examples generate deterministic sample data. The separate benchmark uses measured datasets; its reported scores describe only its stated data and splits.

For scenario-based guidance, see [Use cases](use-cases.md).
