import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from oif.interfaces.ivp import IVP
from scipy import integrate

IMPL_LIST = ["scipy_ode_dopri5", "sundials_cvode", "native_scipy_dopri5"]

RESULT_SOLUTION_FILENAME_TPL = os.path.join("assets", "ivp_burgers_soln_{}.pdf")
RESULT_PERF_FILENAME = os.path.join("assets", "ivp_burgers_perf.pdf")


def _parse_args():
    p = argparse.ArgumentParser()
    subparsers = p.add_subparsers(required=True)
    one_impl = subparsers.add_parser("one")
    one_impl.add_argument(
        "impl",
        choices=IMPL_LIST,
        default=IMPL_LIST[0],
        nargs="?",
    )
    one_impl.set_defaults(func=run_one_impl)

    all_impl = subparsers.add_parser("all")
    all_impl.add_argument(
        "--n_runs", default=1, type=int, help="Number of runs for each implementation"
    )
    all_impl.add_argument(
        "--scalability", action="store_true", help="Run with different resolutions"
    )
    all_impl.set_defaults(func=run_all_impl)

    return p.parse_args()


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

        self._udot = np.empty_like(self.u)

    def compute_rhs(self, __, u: np.ndarray, udot: np.ndarray) -> None:
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

    def compute_rhs_native(self, t, u) -> np.ndarray:
        self.compute_rhs(t, u, self._udot)
        return self._udot


def run_one_impl(args):
    _run_once(args.impl, plot_solution=True)


def run_all_impl(args):
    print("================================================================")
    print("Run all implementations")
    print(f"args.scalability = {args.scalability}")
    print(f"args.n_runs = {args.n_runs}")

    if args.scalability:
        resolutions = [101, 1001, 10_001]
    else:
        resolutions = [1001]
    print(f"Run with resolutions: {resolutions}")

    tts_list = {}
    for N in resolutions:
        tts_list[N] = {}
        for impl in IMPL_LIST:
            tts_list[N][impl] = []

    for N in resolutions:
        for impl in IMPL_LIST:
            for __ in range(args.n_runs):
                elapsed_time = _run_once(impl, N, plot_solution=False)
                tts_list[N][impl].append(elapsed_time)

    print("================================================================")
    print("Statistics:")
    tts_stats = {}
    for impl in IMPL_LIST:
        tts_stats[impl] = {}
        for N in resolutions:
            tts_stats[impl][N] = {}
    for N in resolutions:
        print(f"N = {N}")
        for impl in IMPL_LIST:
            tts_ave = np.mean(tts_list[N][impl])
            tts_std = np.std(tts_list[N][impl], ddof=1)
            print(f"{impl:24s} {tts_ave:6.2f} {tts_std:6.2f}")
            tts_stats[impl][N]["tts_ave"] = tts_ave
            tts_stats[impl][N]["tts_std"] = tts_std

    plt.figure()
    for impl in IMPL_LIST:
        tts_ave = [tts_stats[impl][N]["tts_ave"] for N in resolutions]
        tts_std = [tts_stats[impl][N]["tts_std"] for N in resolutions]
        plt.errorbar(resolutions, tts_ave, yerr=tts_std, label=impl)
    plt.legend(loc="best")
    plt.tight_layout(pad=0.1)
    plt.savefig(RESULT_PERF_FILENAME)


def _run_once(impl, N=1001, plot_solution=True) -> float:
    print("================================================================")
    print(f"Solving Burgers' equation with time integration {impl}")
    begin_time = time.time()
    problem = BurgersEquationProblem(N=N)
    if impl == "native_scipy_dopri5":
        s = integrate.ode(problem.compute_rhs_native)
        s.set_integrator("dopri5", atol=1e-15, rtol=1e-15, nsteps=1000)
        t0 = 0.0
        y0 = problem.u
        s.set_initial_value(y0, t0)
    else:
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
    end_time = time.time()
    elapsed_time = end_time - begin_time
    print("Finished")

    if plot_solution:
        plt.plot(problem.x, soln[0], "--", label="Initial condition")
        plt.plot(problem.x, soln[-1], "-", label="Final solution")
        plt.legend(loc="best")
        plt.savefig(RESULT_SOLUTION_FILENAME_TPL.format(impl))

    return elapsed_time


if __name__ == "__main__":
    args = _parse_args()
    args.func(args)
