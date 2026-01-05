import argparse
import dataclasses
import sys
from typing import Optional, Union

import numpy as np
from openinterfaces.interfaces.optim import Optim


@dataclasses.dataclass
class Args:
    no_artefacts: bool


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--no-artefacts", action="store_true", help="Do not save artifacts")
    args = Args(**vars(p.parse_args()))
    return args


def main(argv=None):
    if argv is None:
        argv = sys.argv
    args = parse_args(argv)

    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

    s = Optim("scipy_optimize")
    s.set_initial_guess(x0)
    s.set_objective_fn(objective_rosen_with_args)

    status, message = s.minimize()

    print(f"Status code: {status}")
    print(f"Solver message: '{message}'")
    print(f"Optimized value: {s.x}")

    if not args.no_artefacts:
        print("Finish")


def objective_rosen_with_args(x, params=(0.5, 1.0)):
    """The Rosenbrock function with additional arguments"""
    a, b = params
    return sum(a * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0) + b


if __name__ == "__main__":
    sys.exit(main())
