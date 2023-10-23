import numpy as np
from oif.core import OIFPyBinding, init_impl


class QeqSolver:
    def __init__(self, impl: str):
        self._binding: OIFPyBinding = init_impl("qeq", impl, 1, 0)

    def solve(self, a: float, b: float, c: float):
        result = np.array([11.0, 22.0])
        self._binding.call("solve_qeq", (a, b, c), (result,))
        return result
