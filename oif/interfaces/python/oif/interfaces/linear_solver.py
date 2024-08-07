import numpy as np
from oif.core import OIFPyBinding, load_impl, unload_impl


class LinearSolver:
    def __init__(self, impl: str):
        self._binding: OIFPyBinding = load_impl("linsolve", impl, 1, 0)

    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        result = np.empty((A.shape[1]))
        self._binding.call("solve_lin", (A, b), (result,))
        return result

    def __del__(self):
        if hasattr(self, "_binding"):
            unload_impl(self._binding)
