import numpy as np
from oif.core import OIFBackend, init_backend


class QeqSolver:
    def __init__(self, provider: str):
        self.backend: OIFBackend = init_backend(provider, "qeq", 1, 0)

    def solve(self, a: float, b: float, c: float):
        result = np.array([11.0, 22.0])
        self.backend.call("solve_qeq", (a, b, c), (result,))
        return result
