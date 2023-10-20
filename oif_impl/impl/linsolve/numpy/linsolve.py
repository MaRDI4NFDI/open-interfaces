import numpy as np
from oif.impl.linsolve import LinsolveInterface


class NumPyLinsolve(LinsolveInterface):
    def solve_lin(self, A, b, result):
        result[:] = np.linalg.solve(A, b)
