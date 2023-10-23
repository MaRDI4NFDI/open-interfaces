import numpy as np
from oif.core import OIFPyBinding, init_impl


class LinearSolver:
    def __init__(self, impl: str):
        self._binding: OIFPyBinding = init_impl("linsolve", impl, 1, 0)

    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        result = np.empty((A.shape[1]))
        self._binding.call("solve_lin", (A, b), (result,))
        return result
