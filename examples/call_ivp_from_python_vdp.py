import argparse
import dataclasses
import os

import matplotlib.pyplot as plt
import numpy as np
from oif.interfaces.ivp import IVP

RESULT_DATA_FILENAME_TPL = "ivp_py_vdp_eq_{:s}.txt"
RESULT_FIG_FILENAME_TPL = "ivp_py_vdp_eq_{:s}.pdf"


@dataclasses.dataclass
class Args:
    impl: str
    outdir: str


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "impl",
        choices=["scipy_ode", "sundials_cvode", "jl_diffeq"],
        default="scipy_ode",
        nargs="?",
    )
    p.add_argument("--outdir", default="assets")
    args = p.parse_args()
    return Args(**vars(args))


class VdPEquationProblem:
    r"""
    Problem class for van der Poll oscillator
    $$
            x''(t) - \mu (1 - x^2) x'(t) + x = 0,
    $$
    with initial condition :math:`x(0) = x'(0) = 0`.
    The 2nd-order equations is transformed to the system of first-order ODEs:
    $$
            y_1'(t) &= y_2, \\
            y_2'(t) &= \mu ( 1 - \left( y_1' \right)^2) y_2 - y_1.
    $$

    Parameters
    ----------
    N : int
        Grid resolution.
    """

    def __init__(self, mu=5.0, tfinal=50.0):
        self.mu = mu

        self.t0 = 0.0
        self.tfinal = tfinal

        self.u0 = [2.0, 0.0]

    def compute_rhs(self, __, u: np.ndarray, udot: np.ndarray, ___) -> None:
        udot[0] = u[1]
        udot[1] = self.mu * (1 - u[0] ** 2) * u[1] - u[0]


def main():
    args = _parse_args()
    impl = args.impl
    print("------------------------------------------------------------------------")
    print("Solving van der Poll oscillator with IVP interface using Python bindings")
    print(f"Implementation: {impl}")
    problem = VdPEquationProblem(mu=1000, tfinal=3000)
    s = IVP(impl)

    t0 = problem.t0
    y0 = problem.u0
    s.set_initial_value(y0, t0)
    s.set_rhs_fn(problem.compute_rhs)
    s.set_tolerances(rtol=1e-8, atol=1e-12)
    if impl == "sundials_cvode":
        s.set_integrator("bdf")
        s.set_integrator("bdf", {"max_num_steps": 30_000})
    elif impl == "scipy_ode":
        # s.set_integrator("dopri5", {"nsteps": 100_000})
        s.set_integrator("vode", {"method": "bdf", "nsteps": 40_000})
    elif impl == "jl_diffeq":
        s.set_integrator("Rosenbrock23")
    else:
        raise ValueError(f"Cannot set integrator for implementation '{impl}'")

    times = np.linspace(problem.t0, problem.tfinal, num=501)

    soln = np.empty((len(times), len(y0)))
    soln[0] = y0
    i = 1
    for t in times[1:]:
        s.integrate(t)
        soln[i] = s.y
        i += 1

    data = np.hstack((times.reshape((-1, 1)), soln))
    data_filename = os.path.join(args.outdir, RESULT_DATA_FILENAME_TPL.format(impl))
    np.savetxt(data_filename, data)

    plt.plot(times, soln[:, 0], "-", label="$y_1$")
    plt.xlabel("Time")
    plt.ylabel("Solution")
    plt.tight_layout(pad=0.1)
    fig_filename = os.path.join(args.outdir, RESULT_FIG_FILENAME_TPL.format(impl))
    plt.savefig(fig_filename.format(impl))
    plt.show()
    print("Finished")


if __name__ == "__main__":
    main()
