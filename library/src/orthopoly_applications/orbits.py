"""Low-rank Weyl-orbit exponential sums in fundamental-weight coordinates."""

from dataclasses import dataclass

import numpy as np


_CARTAN = {
    "A1": ((2,),),
    "A2": ((2, -1), (-1, 2)),
    "B2": ((2, -2), (-1, 2)),
    "G2": ((2, -3), (-1, 2)),
}


def cartan_matrix(type_name: str) -> np.ndarray:
    """Return a supported rank-one or rank-two Cartan matrix."""
    try:
        return np.array(_CARTAN[type_name.upper()], dtype=int)
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"type_name must be one of {sorted(_CARTAN)}") from exc


@dataclass(frozen=True)
class WeylOrbit:
    """Orbit functions for a dominant integral weight.

    ``weight`` is in fundamental-weight coordinates. ``points`` passed to
    ``evaluate`` are dual coordinates, so <weight, point> is a dot product.
    The C sum counts each distinct orbit point once. The S sum is defined
    for strictly dominant weights; at a wall it is identically zero.
    """

    cartan: np.ndarray
    weight: tuple[int, ...]

    def __post_init__(self) -> None:
        C = np.asarray(self.cartan)
        if C.ndim != 2 or C.shape[0] != C.shape[1] or not np.issubdtype(C.dtype, np.integer):
            raise ValueError("cartan must be a square integer matrix")
        if not any(np.array_equal(C, value) for value in map(np.asarray, _CARTAN.values())):
            raise ValueError("supported Cartan matrices: A1, A2, B2, G2")
        w = np.asarray(self.weight)
        if w.shape != (C.shape[0],) or not np.issubdtype(w.dtype, np.integer) or np.any(w < 0):
            raise ValueError("weight must be a nonnegative integral dominant weight")
        object.__setattr__(self, "cartan", C.copy())
        object.__setattr__(self, "weight", tuple(int(v) for v in w))

    def orbit(self) -> tuple[np.ndarray, np.ndarray]:
        """Return distinct orbit weights and reflection-parity signs."""
        C = self.cartan
        start = self.weight
        signs = {start: 1}
        queue = [start]
        for point in queue:
            for i in range(len(start)):
                reflected = tuple(point[j] - point[i] * int(C[i, j]) for j in range(len(start)))
                sign = -signs[point]
                if reflected not in signs:
                    signs[reflected] = sign
                    queue.append(reflected)
        weights = np.array(queue, dtype=int)
        return weights, np.array([signs[v] for v in queue], dtype=int)

    def evaluate(self, points: np.ndarray, kind: str = "C") -> np.ndarray:
        """Evaluate C or S orbit sums at points of shape (..., rank)."""
        if kind not in ("C", "S"):
            raise ValueError("kind must be 'C' or 'S'")
        x = np.asarray(points, dtype=float)
        if x.ndim < 1 or x.shape[-1] != len(self.weight) or not np.all(np.isfinite(x)):
            raise ValueError("points must be finite with final dimension equal to rank")
        if kind == "S" and 0 in self.weight:
            return np.zeros(x.shape[:-1], dtype=complex)
        weights, signs = self.orbit()
        exponentials = np.exp(2j * np.pi * (x @ weights.T))
        return exponentials @ (signs if kind == "S" else np.ones(len(weights), dtype=int))
