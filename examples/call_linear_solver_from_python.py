import argparse

import numpy as np

from oif.interfaces.linear_solver import LinearSolver


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("backend", choices=["c", "python"], nargs="?")
    args = p.parse_args()
    if args.backend is None:
        args.backend = "c"
    return args


def main():
    args = _parse_args()
    backend = args.backend
    print("Calling from Python an open interface for quadratic solver")
    print(f"Backend: {backend}")
    A = np.array(
        [
            [1.0, 1.0],
            [-3.0, 1.0],
        ]
    )
    b = np.array([6.0, 2.0])
    s = LinearSolver(backend)
    x = s.solve(A, b)

    print("Solving system of linear equations:")
    print(f"A={A}")
    print(f"b={b}:")
    print(f"x = {x}")
    print(f"L2 error = {np.linalg.norm(A @ x - b)}")


if __name__ == "__main__":
    main()
