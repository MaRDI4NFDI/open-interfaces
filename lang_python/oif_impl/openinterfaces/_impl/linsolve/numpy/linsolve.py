import numpy as np
from openinterfaces._impl.interfaces.linsolve import LinsolveInterface


class NumPyLinsolve(LinsolveInterface):
    def solve_lin(self, A, b, result):
        if np.isfortran(A):
            print("[linsolve::numpy] Receive matrix in Fortran-order")
        result[:] = np.linalg.solve(A, b)
