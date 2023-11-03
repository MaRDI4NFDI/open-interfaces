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


def rhs(t, y):
    return -y


def main():
    args = _parse_args()
    impl = args.impl
    print("Calling from Python an open interface for quadratic solver")
    print(f"Implementation: {impl}")
    s = IVP(impl)
    s.set_rhs_fn(rhs)
    t0 = 0.0
    y0 = [1.0]
    s.set_initial_value(y0, t0)

    times = np.linspace(t0, t0 + 1, num=11)

    soln = [y0[0]]
    for t in times[1:]:
        s.integrate(t)
        print(f"{t:.3f} {s.y[0]:.6f}")
        soln.append(s.y[0])


if __name__ == "__main__":
    main()
