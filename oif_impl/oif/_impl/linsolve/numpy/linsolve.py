import numpy as np
from oif._impl.interfaces.linsolve import LinsolveInterface


class NumPyLinsolve(LinsolveInterface):
    def solve_lin(self, A, b, result):
        result[:] = np.linalg.solve(A, b)
