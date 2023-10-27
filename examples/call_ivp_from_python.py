import argparse
import dataclasses

import numpy as np
from oif.interfaces.ivp import IVP


@dataclasses.dataclass
class Args:
    impl: str


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "impl",
        choices=["scipy_ode_dopri5"],
        default="scipy_ode_dopri5",
        nargs="?",
    )
    args = p.parse_args()
    return Args(**vars(args))


def rhs(x):
    return -x


def main():
    args = _parse_args()
    impl = args.impl
    print("Calling from Python an open interface for quadratic solver")
    print(f"Implementation: {impl}")
    s = IVP(impl)
    s.set_rhs_fn(rhs)
    t0 = 0.0
    y0 = [1.0]
    s.set_initial_value(t0, y0)

    times = np.linspace(t0, t0 + 1, num=11)

    y = np.empty_like(y0)
    for t in times[1:]:
        s.integrate(t, y)
        print(f"{t} {y}")


if __name__ == "__main__":
    main()
