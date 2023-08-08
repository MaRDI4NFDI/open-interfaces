from oif.core import OIFBackend, init_backend


class QeqSolver:
    def __init__(self, provider: str):
        self.backend: OIFBackend = init_backend("c", "qeq", 1, 0)

    def solve(self, a: float, b: float, c: float):
        res = self.backend.call("solve", (a, b, c), [])
        return res
