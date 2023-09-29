import numpy as np


def solve_lin(A, b, result):
    result[:] = np.linalg.solve(A, b)
