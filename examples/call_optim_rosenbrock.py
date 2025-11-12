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

    x0 = np.array([0.5, 0.6, 0.7])

    s = Optim("scipy_optimize")
    s.set_initial_guess(x0)
    s.set_objective_fn(objective_fn)

    if not args.no_artefacts:
        print("Finish")


def objective_fn(x):
    return np.sum(x**2)


if __name__ == "__main__":
    sys.exit(main())
