import argparse
import dataclasses
import os

import matplotlib.pyplot as plt
import numpy as np
from openinterfaces.interfaces.ivp import IVP


@dataclasses.dataclass
class Args:
    impl: str
    savefig: bool


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "impl",
        choices=["scipy_ode", "sundials_cvode", "jl_diffeq"],
        default="scipy_ode",
        nargs="?",
    )
    p.add_argument(
        "--savefig", "-s", action="store_true", help="Save figure instead of showing it"
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
        self.u0 = 0.5 - 0.25 * np.sin(np.pi * self.x)
        self.invariant = np.sum(np.abs(self.u0))

        self.cfl = 0.5
        self.dt_max = self.dx * self.cfl

        self.t0 = 0.0
        self.tfinal = 2.0

    def compute_rhs(self, __, u: np.ndarray, udot: np.ndarray, ___) -> int:
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
        return 0


def main():
    args = _parse_args()
    impl = args.impl
    print("Solving Burgers' equation with IVP interface using Python bindings")
    print(f"Implementation: {impl}")
    problem = BurgersEquationProblem(N=1001)
    s = IVP(impl)
    s.set_initial_value(problem.u0, problem.t0)
    s.set_rhs_fn(problem.compute_rhs)

    times = np.linspace(problem.t0, problem.tfinal, num=11)

    soln = [problem.u0]
    for t in times[1:]:
        s.integrate(t)
        soln.append(s.y)

    plt.plot(problem.x, soln[0], "--", label="Initial condition")
    plt.plot(problem.x, soln[-1], "-", label="Final solution")
    plt.xlabel(r"$x$")
    plt.ylabel(r"Solution of Burgers' equation")
    plt.legend(loc="best")
    plt.tight_layout(pad=0.1)

    if args.savefig:
        plt.savefig(os.path.join("assets", f"ivp_py_burgers_eq_{impl}.pdf"))
    else:
        plt.show()


if __name__ == "__main__":
    main()
