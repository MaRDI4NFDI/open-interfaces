import argparse
import dataclasses
import os

import matplotlib.pyplot as plt
import numpy as np
from oif.interfaces.ivp import IVP


@dataclasses.dataclass
class Args:
    impl: str


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "impl",
        choices=["scipy_ode", "sundials_cvode", "jl_diffeq"],
        default="scipy_ode",
        nargs="?",
    )
    args = p.parse_args()
    return Args(**vars(args))


class BurgersEquationProblem:
    r"""
    Problem class for inviscid Burgers' equation:
    $$
            u_t + 0.5 * (u^2)_x = 0, \quad x \in [0, 2], \quad t \in [0, 2]
    $$
    with initial condition :math:`u(x, 0) = 0.5 - 0.25 * sin(pi * x)`
    and periodic boundary conditions.

    Parameters
    ----------
    N : int
        Grid resolution.
    """

    def __init__(self, N=101):
        self.N = N

        self.x, self.dx = np.linspace(0, 2, num=N, retstep=True)
        self.u = 0.5 - 0.25 * np.sin(np.pi * self.x)
        self.invariant = np.sum(np.abs(self.u))

        self.cfl = 0.5
        self.dt_max = self.dx * self.cfl

        self.t0 = 0.0
        self.tfinal = 2.0

    def compute_rhs(self, __, u: np.ndarray, udot: np.ndarray, ___) -> None:
        dx = self.dx

        f = 0.5 * u**2
        local_ss = np.maximum(np.abs(u[0:-1]), np.abs(u[1:]))
        local_ss = np.max(np.abs(u))
        f_hat = 0.5 * (f[0:-1] + f[1:]) - 0.5 * local_ss * (u[1:] - u[0:-1])
        f_plus = f_hat[1:]
        f_minus = f_hat[0:-1]
        udot[1:-1] = -1.0 / dx * (f_plus - f_minus)

        local_ss_rb = np.maximum(np.abs(u[0]), np.abs(u[-1]))
        f_rb = 0.5 * (f[0] + f[-1]) - 0.5 * local_ss_rb * (u[0] - u[-1])
        f_lb = f_rb

        udot[+0] = -1.0 / dx * (f_minus[0] - f_lb)
        udot[-1] = -1.0 / dx * (f_rb - f_plus[-1])


def main():
    args = _parse_args()
    impl = args.impl
    print("Solving Burgers' equation with IVP interface using Python bindings")
    print(f"Implementation: {impl}")
    problem = BurgersEquationProblem(N=1001)
    s = IVP(impl)
    t0 = 0.0
    y0 = problem.u
    s.set_initial_value(y0, t0)
    s.set_rhs_fn(problem.compute_rhs)

    times = np.linspace(problem.t0, problem.tfinal, num=11)
    times = np.arange(problem.t0, problem.tfinal + problem.dt_max, step=problem.dt_max)

    soln = [y0]
    for t in times[1:]:
        s.integrate(t)
        soln.append(s.y)

    plt.plot(problem.x, soln[0], "--", label="Initial condition")
    plt.plot(problem.x, soln[-1], "-", label="Final solution")
    plt.legend(loc="best")
    plt.tight_layout(pad=0.1)
    plt.savefig(os.path.join("assets", f"ivp_burgers_soln_{impl}.pdf"))
    print("Finished")


if __name__ == "__main__":
    main()
