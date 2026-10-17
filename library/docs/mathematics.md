# Mathematical conventions and scope

## Classical families

The package follows standard polynomial, rather than orthonormal, conventions:

| Family | Domain | Weight | Representative identity |
| --- | --- | --- | --- |
| Jacobi `P_n^(α,β)` | `[-1, 1]` | `(1-x)^α(1+x)^β`, `α,β > -1` | General two-parameter family |
| Hermite `H_n` | real line | `exp(-x²)` | Physicists' normalization |
| Probabilists' Hermite `He_n` | real line | `exp(-x²/2)` | `He_2(x)=x²-1` |
| Chebyshev `T_n` | `[-1, 1]` | `(1-x²)^(-1/2)` | `T_n(cos θ)=cos(nθ)` |
| Laguerre `L_n` | `[0, ∞)` | `exp(-x)` | `L_1(x)=1-x` |

`project_function` computes `c_n = <f, P_n> / ||P_n||²` using Gaussian quadrature. It is a weighted approximation to the mathematical projection when `f` is not a low-degree polynomial.

## Graph filtering

For normalized graph Laplacian `L`, all eigenvalues lie in `[0, 2]`. The trainable graph classifier constructs columns from `T_k(L-I)X`. This uses only sparse/dense matrix products and Chebyshev recurrence; it does not decompose the full graph spectrum. The classifier minimizes softmax cross-entropy with ridge penalty on non-bias weights.

## Laguerre history

The basis for a nonnegative lag `t` is `φ_n(t) = sqrt(2a) exp(-at) L_n(2at)` with `a = rate > 0`. These functions are orthonormal on `[0, ∞)` in the continuous inner product. `LaguerreExpansion` fits sampled values by least squares on a finite interval. `LaguerreForecaster` encodes reversed history windows with the sampled basis and learns a future-value readout; it is a forecasting model, not a recurrent neural network or a dynamical-system reduction algorithm.

## Weyl orbit sums

For a dominant weight `λ`, the C sum contains one term `exp(2πi <μ,x>)` per distinct orbit point `μ`. For a strictly dominant weight, the S sum also multiplies each term by the determinant sign of its Weyl element. For A1 and weight 1, `C_1(x)=2cos(2πx)` and `S_1(x)=2i sin(2πx)` under this exponential convention.

The implementation covers A1, A2, B2, and G2 only. It does not claim general continuous or discrete Weyl-transform orthogonality, character computations, or root-lattice classification. The coursework's Jacobi integral is a squared norm with a Kronecker delta, so the package names it `norm_squared` rather than calling the standard polynomial basis orthonormal.
