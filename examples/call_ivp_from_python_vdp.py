import argparse
import dataclasses
import os

import matplotlib.pyplot as plt
import numpy as np
from openinterfaces.interfaces.ivp import IVP

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
        choices=[
            "scipy_ode-dopri5",
            "scipy_ode-dopri5-100k",
            "scipy_ode-vode",
            "scipy_ode-vode-40k",
            "sundials_cvode",
            "jl_diffeq-rosenbrock23",
        ],
        default="scipy_ode-dopri5",
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
    The 2nd-order equation is transformed to the system of first-order ODEs:
    $$
            y_1'(t) &= y_2, \\
            y_2'(t) &= \mu ( 1 - \left( y_1' \right)^2) y_2 - y_1.
    $$

    Parameters
    ----------
    mu : float
        Parameter that determines the stiffness of the problem.
    """

    def __init__(self, mu=5.0):
        self.mu = mu

    def compute_rhs(self, __, y, ydot, ___):
        ydot[0] = y[1]
        ydot[1] = self.mu * (1 - y[0] ** 2) * y[1] - y[0]
        return 0


def main():
    args = _parse_args()
    impl = args.impl
    impl, integrator = args.impl.split("-", 1)
    print("------------------------------------------------------------------------")
    print("Solving van der Poll oscillator with IVP interface using Python bindings")
    print(f"Implementation: {impl}")
    if integrator:
        print(f"Integrator options: {integrator}")

    problem = VdPEquationProblem(mu=1000)

    y0 = [2.0, 0.0]  # Initial condition
    t0 = 0  # Initial time
    tfinal = 3000  # Final time

    solver = IVP(impl)
    solver.set_initial_value(y0, t0)
    solver.set_rhs_fn(problem.compute_rhs)
    solver.set_tolerances(rtol=1e-8, atol=1e-12)
    if impl == "sundials_cvode":
        solver.set_integrator("bdf")
        solver.set_integrator("bdf", {"max_num_steps": 30_000})
    elif impl == "scipy_ode" and integrator == "dopri5":
        solver.set_integrator("dopri5")  # It is already the default integrator
    elif impl == "scipy_ode" and integrator == "dopri5-100k":
        solver.set_integrator("dopri5", {"nsteps": 100_000})
    elif impl == "scipy_ode" and integrator == "vode":
        solver.set_integrator("vode", {"method": "bdf"})
    elif impl == "scipy_ode" and integrator == "vode-40k":
        solver.set_integrator("vode", {"method": "bdf", "nsteps": 40_000})
    elif impl == "jl_diffeq" and integrator.lower() == "Rosenbrock23".lower():
        solver.set_integrator(
            "Rosenbrock23",
            {
                "autodiff": False,
            },
        )
    else:
        raise ValueError(f"Cannot set integrator for implementation '{impl}'")

    times = np.linspace(t0, tfinal, num=501)

    solution = np.empty((len(times), len(y0)))
    solution[0] = y0
    i = 1
    for t in times[1:]:
        solver.integrate(t)
        solution[i] = solver.y
        i += 1

    data = np.hstack((times.reshape((-1, 1)), solution))
    data_filename = os.path.join(args.outdir, RESULT_DATA_FILENAME_TPL.format(impl))
    np.savetxt(data_filename, data)

    plt.plot(times, solution[:, 0], "-", label="$y_1$")
    plt.xlabel("Time")
    plt.ylabel("Solution")
    plt.tight_layout(pad=0.1)
    fig_filename = os.path.join(args.outdir, RESULT_FIG_FILENAME_TPL.format(impl))
    plt.savefig(fig_filename.format(impl))
    plt.show()
    print("Finished")


if __name__ == "__main__":
    main()
