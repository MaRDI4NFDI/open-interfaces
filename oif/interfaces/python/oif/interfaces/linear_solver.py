import numpy as np
from oif.core import OIFBackend, init_backend


class LinearSolver:
    def __init__(self, impl: str):
        self.backend: OIFBackend = init_backend("linsolve", impl, 1, 0)

    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        result = np.empty((A.shape[1]))
        self.backend.call("solve_lin", (A, b), (result,))
        return result
