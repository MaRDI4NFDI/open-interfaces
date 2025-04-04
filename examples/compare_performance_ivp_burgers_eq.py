import argparse
import os
import pickle
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from oif.interfaces.ivp import IVP
from oif.util import UsedMemoryMonitor
from scipy import integrate

IMPL_LIST = ["scipy_ode_dopri5", "sundials_cvode", "native_scipy_ode_dopri5"]
RESOLUTIONS = [101, 201, 401, 801, 1001, 2001, 4001, 8001, 10_001, 20_001, 40_001]

RESULT_SOLUTION_FILENAME_TPL = os.path.join("assets", "ivp_burgers_soln_{}.pdf")
RESULT_PERF_FILENAME = os.path.join("assets", "ivp_burgers_perf.pdf")
RESULT_DATA_PICKLE = os.path.join("assets", "ivp_burgers_data.pickle")
RESULT_PERF_NORMALIZED_FILENAME = os.path.join(
    "assets", "ivp_burgers_perf_normalized.pdf"
)

current_datetime = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
memory_monitor = UsedMemoryMonitor(csv_filename=f"memory_usage_{current_datetime}.csv")


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
    print(f"args.n_runs = {args.n_runs}")

    print(f"Run with resolutions: {RESOLUTIONS}")

    tts_list = {}
    for impl in IMPL_LIST:
        tts_list[impl] = {}
        for N in RESOLUTIONS:
            tts_list[impl][N] = []

    for impl in IMPL_LIST:
        for N in RESOLUTIONS:
            for __ in range(args.n_runs):
                elapsed_time = _run_once(impl, N, plot_solution=False)
                tts_list[impl][N].append(elapsed_time)

    analyze(tts_list)

    with open(RESULT_DATA_PICKLE, "wb") as fh:
        pickle.dump(tts_list, fh)

    return tts_list


def analyze(tts_list):
    print("================================================================")
    print("Statistics:")
    tts_stats = {}
    for impl in IMPL_LIST:
        tts_stats[impl] = {}
        for N in RESOLUTIONS:
            tts_stats[impl][N] = {}

    for impl in IMPL_LIST:
        for N in RESOLUTIONS:
            tts_ave = np.mean(tts_list[impl][N])
            tts_std = np.std(tts_list[impl][N], ddof=1)
            tts_stats[impl][N]["tts_ave"] = tts_ave
            tts_stats[impl][N]["tts_std"] = tts_std

    col_sep = 5 * " "
    print(
        "{:^24s} ".format("N")
        + col_sep.join(["{:^13d}".format(N) for N in RESOLUTIONS])
    )
    for impl in IMPL_LIST:
        print(
            "{:24s} ".format(impl)
            + col_sep.join(
                [
                    "{:6.2f} {:6.2f}".format(
                        tts_stats[impl][N]["tts_ave"], tts_stats[impl][N]["tts_std"]
                    )
                    for N in RESOLUTIONS
                ]
            )
        )

    plt.figure()
    for impl in IMPL_LIST:
        tts_ave = [tts_stats[impl][N]["tts_ave"] for N in RESOLUTIONS]
        tts_std = [tts_stats[impl][N]["tts_std"] for N in RESOLUTIONS]
        plt.errorbar(RESOLUTIONS, tts_ave, fmt="-o", yerr=tts_std, label=impl)
    plt.legend(loc="best")
    plt.tight_layout(pad=0.1)
    plt.savefig(RESULT_PERF_FILENAME)

    # Plot relative times (normalized by native performance).
    impl = IMPL_LIST[-1]
    assert impl.startswith("native_")
    tts_ave_native = np.array([tts_stats[impl][N]["tts_ave"] for N in RESOLUTIONS])
    tts_std_native = np.array([tts_stats[impl][N]["tts_std"] for N in RESOLUTIONS])
    plt.figure()
    for impl in IMPL_LIST[:-1]:
        tts_ave = np.array([tts_stats[impl][N]["tts_ave"] for N in RESOLUTIONS])
        tts_std = np.array([tts_stats[impl][N]["tts_std"] for N in RESOLUTIONS])
        tts_std_normalized = np.sqrt(
            np.square(tts_std / tts_ave) + np.square(tts_std_native / tts_ave_native)
        )
        plt.errorbar(
            RESOLUTIONS,
            tts_ave / tts_ave_native,
            yerr=tts_std_normalized,
            fmt="-o",
            label=impl,
        )
    plt.legend(loc="best")
    plt.tight_layout(pad=0.1)
    plt.savefig(RESULT_PERF_NORMALIZED_FILENAME)


def _run_once(impl, N=1001, plot_solution=True) -> float:
    print("================================================================")
    print(f"Solving Burgers' equation with time integration {impl}, N = {N}")
    begin_time = time.time()
    problem = BurgersEquationProblem(N=N)
    if impl == "native_scipy_ode_dopri5":
        print(f"Use native impl {impl}")
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

    times = np.arange(problem.t0, problem.tfinal + problem.dt_max, step=problem.dt_max)

    soln = [y0]
    for i, t in enumerate(times[1:]):
        s.integrate(t)
        if i % 100 == 0:
            memory_monitor.record()
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
