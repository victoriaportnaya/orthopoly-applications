"""Chebyshev spectral filtering without a Laplacian eigendecomposition."""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


def normalized_laplacian(adjacency: np.ndarray | sparse.spmatrix):
    """Return I - D^(-1/2) A D^(-1/2) for an undirected weighted graph.

    Isolated vertices receive a zero diagonal entry, following the usual
    normalized-Laplacian convention. The output preserves sparse input.
    """
    if sparse.issparse(adjacency):
        a = adjacency.tocsr().astype(float)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("adjacency must be square")
        if a.data.size and (not np.all(np.isfinite(a.data)) or np.any(a.data < 0)):
            raise ValueError("adjacency weights must be finite and nonnegative")
        if (a - a.T).nnz and not np.allclose((a - a.T).data, 0):
            raise ValueError("adjacency must be symmetric")
        degrees = np.asarray(a.sum(axis=1)).ravel()
        invsqrt = np.zeros_like(degrees)
        np.divide(1, np.sqrt(degrees), out=invsqrt, where=degrees > 0)
        d = sparse.diags(invsqrt)
        return sparse.diags((degrees > 0).astype(float)) - d @ a @ d
    a = np.asarray(adjacency, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("adjacency must be a finite nonnegative square matrix")
    if not np.allclose(a, a.T):
        raise ValueError("adjacency must be symmetric")
    degrees = a.sum(axis=1)
    invsqrt = np.zeros_like(degrees)
    np.divide(1, np.sqrt(degrees), out=invsqrt, where=degrees > 0)
    return np.diag((degrees > 0).astype(float)) - invsqrt[:, None] * a * invsqrt[None, :]


def chebyshev_graph_filter(
    laplacian: np.ndarray | sparse.spmatrix,
    features: np.ndarray,
    coefficients: np.ndarray,
    *,
    lambda_max: float | None = None,
) -> np.ndarray:
    """Apply sum_k coefficients[k] T_k(2L/lambda_max - I) to node features.

    ``features`` may have shape (nodes,) or (nodes, channels). For large sparse
    graphs, pass a known spectral upper bound (2 for a normalized Laplacian)
    to avoid computing an extreme eigenvalue. Coefficients use the literal
    Chebyshev series convention: the degree-zero term is not halved.
    """
    L = laplacian.tocsr().astype(float) if sparse.issparse(laplacian) else np.asarray(laplacian, dtype=float)
    x = np.asarray(features, dtype=float)
    c = np.asarray(coefficients, dtype=float)
    if L.ndim != 2 or L.shape[0] != L.shape[1] or x.ndim not in (1, 2) or x.shape[0] != L.shape[0]:
        raise ValueError("laplacian must be square and match the feature rows")
    if c.ndim != 1 or len(c) == 0 or not np.all(np.isfinite(c)) or not np.all(np.isfinite(x)):
        raise ValueError("features and coefficients must be finite; coefficients must be nonempty")
    if sparse.issparse(L):
        if L.data.size and not np.all(np.isfinite(L.data)):
            raise ValueError("laplacian must be finite")
    elif not np.all(np.isfinite(L)):
        raise ValueError("laplacian must be finite")
    if lambda_max is None:
        if L.shape[0] == 1:
            lambda_max = float(L[0, 0])
        elif sparse.issparse(L):
            lambda_max = float(sparse_linalg.eigsh(L, k=1, which="LA", return_eigenvectors=False)[0])
        else:
            lambda_max = float(np.linalg.eigvalsh(L)[-1])
    if not np.isfinite(lambda_max) or lambda_max <= 0:
        raise ValueError("lambda_max must be positive; a zero Laplacian has no spectral scale")

    def scaled_matvec(v):
        return (2 / lambda_max) * (L @ v) - v

    t0 = x
    result = c[0] * t0
    if len(c) == 1:
        return result
    t1 = scaled_matvec(x)
    result = result + c[1] * t1
    for weight in c[2:]:
        t0, t1 = t1, 2 * scaled_matvec(t1) - t0
        result = result + weight * t1
    return result
