import argparse
import dataclasses

import numpy as np
from oif.interfaces.linear_solver import LinearSolver


@dataclasses.dataclass
class Args:
    impl: str


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "impl", type=str, choices=["c_lapack", "python"], default="c_lapack", nargs="?"
    )
    args = p.parse_args()
    return Args(**vars(args))


def main():
    args = _parse_args()
    impl = args.impl
    print("Calling from Python an open interface for linear algebraic systems")
    print(f"Implementation: {impl}")
    A = np.array(
        [
            [1.0, 1.0],
            [-3.0, 1.0],
        ]
    )
    b = np.array([6.0, 2.0])
    s = LinearSolver(impl)
    x = s.solve(A, b)

    print("Solving system of linear equations:")
    print(f"A={A}")
    print(f"b={b}:")
    print(f"x = {x}")
    print(f"L2 error = {np.linalg.norm(A @ x - b)}")


if __name__ == "__main__":
    main()
