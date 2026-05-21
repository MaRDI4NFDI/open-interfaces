#!/usr/bin/env python3
import argparse
import sys

import numpy as np
from openinterfaces.interfaces.optim import Optim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("impl", nargs="?", default="optim_jl")
    parser.add_argument("method", nargs="?", default="NelderMead")
    parser.add_argument("linesearch", nargs="?", default="StrongWolfe")
    return parser.parse_args()


def rosenbrock_objective_fn(x, a):
    """The Rosenbrock function with parameter"""
    return np.sum(a * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1.0 - x[:-1]) ** 2.0)


def rosenbrock_grad_fn(x, grad_f, a):
    """The gradient of the Rosenbrock function"""
    xi = x[:-1]
    xip1 = x[1:]

    grad_f[:-1] = -4.0 * a * xi * (xip1 - xi**2.0) - 2.0 * (1.0 - xi)
    grad_f[-1] = 0.0
    grad_f[1:] += 2.0 * a * (xip1 - xi**2.0)
    return 0


def main():
    args = parse_args()
    impl = args.impl
    method = args.method
    linesearch = args.linesearch

    print("Calling from Python an open interface for optimization")
    print(f"Implementation: {impl}")
    print(f"Method: {method}")
    print(f"Line search (only for optim_jl:BFGS): {linesearch}")

    x0 = np.array([3.14, 2.72, 6.18, 9.81, 8.31])
    user_data = 10.0

    s = Optim(impl)
    s.set_initial_guess(x0)
    s.set_user_data(user_data)

    if impl == "scipy_optimize":
        if method == "NelderMead":
            s.set_method("nelder-mead", {"fatol": 1e-11})
        elif method == "BFGS":
            s.set_method("BFGS", {"gtol": 1e-8})
        else:
            raise ValueError(f"Unsupported method '{method}'")
    elif impl == "optim_jl":
        if method == "NelderMead":
            s.set_method("NelderMead", {"g_abstol": 1e-11})
        elif method == "BFGS":
            s.set_method("BFGS", {"g_abstol": 1e-8, "linesearch": linesearch})
        else:
            raise ValueError(f"Unsupported method '{method}'")
    else:
        raise ValueError("Unknown implementation")

    s.set_objective_fn(rosenbrock_objective_fn)
    s.set_grad_fn(rosenbrock_grad_fn)

    status, message = s.minimize()
    x = s.x

    print(f"Message: {message}")
    assert status == 0
    print(f"x = {x}")
    if np.all(np.abs(x - 1.0) < 1e-5):  # The solution is [1, 1, ..., 1].
        print("\033[1;32mSUCCESS\033[0m Found solution is close to the exact one")
    else:
        print("\033[1;31mFAIL\033[0m Found solution is NOT close to the exact one")


if __name__ == "__main__":
    sys.exit(main())
