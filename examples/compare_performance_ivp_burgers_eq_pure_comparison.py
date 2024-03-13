#!/usr/bin/env python
"""Compare 'native' call to SciPy Dormand--Prince ODE solver and the call
via an extra C-wrapper function."""
import argparse
import ctypes
import os
import pickle
import time
from typing import TypeAlias, Union

import matplotlib.pyplot as plt
import numpy as np
from line_profiler import profile
from scipy import integrate

CtypesType: TypeAlias = Union[ctypes._SimpleCData, ctypes._Pointer]

OIF_INT = 1
OIF_FLOAT64 = 3
OIF_ARRAY_F64 = 5


class OIFArrayF64(ctypes.Structure):
    _fields_ = [
        ("nd", ctypes.c_int),
        ("dimensions", ctypes.POINTER(ctypes.c_long)),
        ("data", ctypes.POINTER(ctypes.c_double)),
    ]


IMPL_LIST = ["scipy_ode_dopri5", "native_scipy_ode_dopri5"]
RESOLUTIONS = [101, 1001, 10_001]

RESULT_SOLUTION_FILENAME_TPL = os.path.join(
    "assets", "ivp_burgers_soln_{}_pure_comparison.pdf"
)
RESULT_PERF_FILENAME = os.path.join("assets", "ivp_burgers_perf_pure_comparison.pdf")
RESULT_DATA_PICKLE = os.path.join("assets", "ivp_burgers_data_pure_comparison.pickle")


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


class TwoConversions:
    """Add conversions between arrays: NumPy -> OIFArrayF64 -> NumPy"""

    def __init__(self, problem):
        self.problem = problem
        # Hardcode is harder than softcode.
        self.arg_types = [OIF_FLOAT64, OIF_ARRAY_F64, OIF_ARRAY_F64]

    def compute(self, t: float, u: np.ndarray) -> np.ndarray:
        c_args = self._c_args_from_py_args(t, u)
        py_args = self._py_args_from_c_args(*c_args)
        begin = time.time()
        result = self.problem.compute_rhs_native(*py_args)
        end = time.time()
        print("{:35s} {:.2e}".format("Elapsed time compute_rhs_native", end - begin))
        return result

    @profile
    def _c_args_from_py_args(self, *args) -> list:
        begin = time.time()
        c_args: list[CtypesType] = []
        for i, (t, v) in enumerate(zip(self.arg_types, args)):
            if t == OIF_INT:
                c_args.append(ctypes.c_int(v))
            elif t == OIF_FLOAT64:
                c_args.append(ctypes.c_double(v))
            elif t == OIF_ARRAY_F64:
                assert v.dtype == np.float64
                nd = v.ndim
                dimensions = (ctypes.c_long * len(v.shape))(*v.shape)
                double_p_t = ctypes.POINTER(ctypes.c_double)
                data = v.ctypes.data_as(double_p_t)

                oif_array = OIFArrayF64(nd, dimensions, data)
                c_args.append(ctypes.pointer(oif_array))
        end = time.time()
        print("{:35s} {:.2e}".format("Elapsed time _c_args_from_py_args", end - begin))
        return c_args

    @profile
    def _py_args_from_c_args(self, *args) -> list:
        begin = time.time()
        py_args = []
        py_args.append(args[0])
        v = args[1]
        arr_type = ctypes.c_double * v.contents.dimensions[0]
        py_args.append(
            np.ctypeslib.as_array(
                arr_type.from_address(ctypes.addressof(v.contents.data.contents))
            )
        )
        # v = args[2]
        # py_args.append(
        #     np.ctypeslib.as_array(
        #         arr_type.from_address(ctypes.addressof(v.contents.data.contents))
        #     )
        # )
        end = time.time()
        print("{:35s} {:.2e}".format("Elapsed time _py_args_from_c_args", end - begin))

        return py_args


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
                        tts_stats[impl][N]["tts_ave"], tts_stats[impl][N]["tts_ave"]
                    )
                    for N in RESOLUTIONS
                ]
            )
        )

    plt.figure()
    for impl in IMPL_LIST:
        tts_ave = [tts_stats[impl][N]["tts_ave"] for N in RESOLUTIONS]
        tts_std = [tts_stats[impl][N]["tts_std"] for N in RESOLUTIONS]
        plt.errorbar(RESOLUTIONS, tts_ave, yerr=tts_std, label=impl)
    plt.legend(loc="best")
    plt.tight_layout(pad=0.1)
    plt.savefig(RESULT_PERF_FILENAME)


def _run_once(impl, N=1001, plot_solution=True) -> float:
    print("================================================================")
    print(f"Solving Burgers' equation with time integration {impl}")
    begin_time = time.time()
    problem = BurgersEquationProblem(N=N)
    if impl == "native_scipy_ode_dopri5":
        print(f"Use native impl {impl}")
        s = integrate.ode(problem.compute_rhs_native)
    else:
        two_conversions = TwoConversions(problem)
        s = integrate.ode(two_conversions.compute)
    s.set_integrator("dopri5", atol=1e-15, rtol=1e-15, nsteps=1000)
    t0 = 0.0
    y0 = problem.u
    s.set_initial_value(y0, t0)

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
