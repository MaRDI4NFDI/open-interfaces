import numpy as np

from oif.core import OIFBackend, init_backend


class LinearSolver:
    def __init__(self, provider: str):
        self.backend: OIFBackend = init_backend(provider, "linear_solver", 1, 0)

    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        result = np.empty((A.shape[1]))
        self.backend.call("solve_lin", (A, b), (result,))
        return result
