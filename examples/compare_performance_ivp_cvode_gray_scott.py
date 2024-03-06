import argparse
import os
import pickle
import sys
import time

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from oif.interfaces.ivp import IVP
from oif.util import Laplacian2DApproximator
from scikits.odes.ode import ode

IMPL_LIST = ["sundials_cvode", "native_sundials_cvode", "forward_euler"]
RESOLUTIONS = [64, 128, 256, 512]

RESULT_SOLUTION_FILENAME_TPL = os.path.join("assets", "ivp_cvode_gs_soln_{}.pdf")
RESULT_PERF_FILENAME = os.path.join("assets", "ivp_cvode_gs_perf.pdf")
RESULT_DATA_PICKLE = os.path.join("assets", "ivp_cvode_gs_data.pickle")
RESULT_PERF_NORMALIZED_FILENAME = os.path.join(
    "assets", "ivp_cvode_gs_perf_normalized.pdf"
)


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
    one_impl.add_argument(
        "--anim-play", action="store_true", help="Play animation during computation"
    )
    one_impl.add_argument(
        "--anim-save",
        action="store_true",
        help="Save animation to file after compution",
    )
    one_impl.set_defaults(func=run_one_impl)

    all_impl = subparsers.add_parser("all")
    all_impl.add_argument(
        "--n_runs", default=1, type=int, help="Number of runs for each implementation"
    )
    all_impl.set_defaults(func=run_all_impl)

    return p.parse_args()


class GrayScottProblem:
    r"""
    Problem class for 2D Gray--Scott reaction-diffusion system:
    $$
        \frac{\partial u}{\partial t} = D_u \Delta u - uv^2 + F(1-U), \\
        \frac{\partial v}{\partial t} = D_v \Delta v + uv^2 - (F + k) v
     $$
    with periodic boundary conditions and initial condition described in [1_].
    on domain $[-2.5, 2.5]^2$.
    Parameters of the problem are F, k, D_u, D_v.

    Parameters
    ----------
    N : int
        Grid resolution.
    seed : int, optional
        Random seed to make results reproducible.
    left, right, bottom, top : float, optional
        Coordinates of the boundary points of the physical 2D domain.

    References
    ----------
    [1_]: Pearson J. E. Complex patterns in a simple system. Science, 1993,
    vol. 261.
    """

    def __init__(self, N=256, seed=None, left=-2.5, right=2.5, bottom=-2.5, top=2.5):
        self._count = 0

        # Problem parameters
        self.F = 0.055
        self.k = 0.062
        self.Du = 2 * 10**-5
        self.Dv = 10**-5

        # Resolution in space and time.
        self.x, self.dx = np.linspace(-2.5, +2.5, num=N + 1, retstep=True)
        self.dy = self.dx
        self.N = N + 1
        self.t0 = 0.0
        self.tfinal = 1_000.0
        self.dt = 1.0

        self.extent = [left, right, bottom, top]

        # Solution is initialized to a random field of uniform noise.
        U = np.ones((self.N, self.N))
        V = np.zeros((self.N, self.N))

        s = N // 2
        U[s - 20 : s + 20, s - 20 : s + 20] = 0.50
        V[s - 20 : s + 20, s - 20 : s + 20] = 0.25

        if seed is not None:
            np.random.seed(seed)
        U[s - 20 : s + 20, s - 20 : s + 20] += 0.1 * np.random.rand(40, 40)
        V[s - 20 : s + 20, s - 20 : s + 20] += 0.1 * np.random.rand(40, 40)

        # Solution state vector for time integrators.
        self.y0 = np.hstack((np.reshape(U, (-1,)), np.reshape(V, (-1,))))

        # Right-hand side vector.
        self._udot = np.empty_like(self.y0)
        self.approx = Laplacian2DApproximator(self.N, self.dx, self.dy)

    def compute_rhs(self, t, y: np.ndarray, ydot: np.ndarray) -> None:
        s = len(y) // 2
        N = self.N
        assert len(y) == 2 * N**2
        F, k, Du, Dv = self.F, self.k, self.Du, self.Dv
        U = np.reshape(y[:s], (N, N))
        V = np.reshape(y[s:], (N, N))
        assert U.base is y
        assert V.base is y

        # We compute the Laplacian of u and v.
        deltaU = self.approx.laplacian_periodic(U)
        deltaV = self.approx.laplacian_periodic(V)

        # plt.ioff()
        # plt.figure()
        # plt.imshow(V)
        # plt.colorbar()
        # plt.title(f"V, time = {t}")
        # plt.tight_layout(pad=0.1)
        # plt.show()
        Udot = np.reshape(ydot[:s], (N, N))
        Vdot = np.reshape(ydot[s:], (N, N))
        Udot[:] = Du * deltaU - U * V**2 + F * (1 - U)
        Vdot[:] = Dv * deltaV + U * V**2 - (F + k) * V
        assert Udot.base is ydot
        assert Vdot.base is ydot
        # plt.figure()
        # plt.imshow(Vdot)
        # plt.colorbar()
        # plt.title("Vdot")
        # plt.tight_layout(pad=0.1)
        # plt.show()

    def compute_rhs_native(self, t):
        self.compute_rhs(t, self.y, self._udot)
        return self._udot

    def plot_2D_solution(
        self,
        y: np.ndarray,
        fig=None,
        im=None,
        cbar=None,
        title=None,
    ):
        assert y.ndim == 1  # It is a 1D array from a time-integration solver.
        s = len(y) // 2
        U = np.reshape(y[:s], (self.N, self.N))
        # V = np.reshape(y[s:], (self.N, self.N))

        umin = np.min(U)
        umax = np.max(U)
        Uplot = (U - umin) / (umax - umin)
        Uplot = U

        R = 0
        G = 155
        B = 210

        #         Uplot_RGB = np.empty((self.N, self.N, 3))
        #         Uplot_RGB[:, :, 0] = (1 - Uplot) * R
        #         Uplot_RGB[:, :, 1] = Uplot * G
        #         Uplot_RGB[:, :, 2] = Uplot * B

        if fig is None:
            fig = plt.figure()
        ax = plt.gca()

        if im is None:
            im = plt.imshow(
                Uplot,
                cmap=mpl.colormaps["viridis"],
                interpolation="bilinear",
                extent=self.extent,
            )
        else:
            im.set_data(Uplot)
        if cbar is None:
            cbar = plt.colorbar()
        else:
            im.set_clim(umin, umax)

        if title is not None:
            ax.set_title(title)

        return fig, im, cbar


class AnimatedSolution:
    def __init__(self, problem, solver, times):
        self.problem = problem
        self.fig, self.im, self.cbar = problem.plot_2D_solution(
            problem.y0, title=f"Time {0:.3f}"
        )
        self.fig.set_size_inches(12.8, 9.6)
        self.ax = plt.gca()
        plt.tight_layout(pad=0.1)
        self.solver = solver

        self.fig.canvas.mpl_connect("key_release_event", self.on_key_release)
        if mpl.get_backend() == "macosx":
            print(
                "WARNING: Matplotlib 'macosx' backend cannot detect "
                "'Ctrl-C' key combination"
            )
        self._stop = False

        self.fps = 60
        self.times = times

        self.anim = animation.FuncAnimation(
            self.fig,
            self.update,
            init_func=self.init_func,
            frames=times,
            interval=1000.0 / float(self.fps),
            repeat=False,
            blit=True,
        )
        # plt.show()

    def init_func(self):
        return (self.im,)

    def update(self, i):
        print(i)
        if self._stop is True:
            raise KeyboardInterrupt()
        if self.fig != plt.gcf():
            raise KeyboardInterrupt()
        t = i
        self.solver.integrate(t)
        self.problem.plot_2D_solution(
            self.solver.y,
            fig=self.fig,
            im=self.im,
            cbar=self.cbar,
            title=f"Time {t:.3f}",
        )

        return (self.im,)

    def on_key_release(self, event):
        """Interrupt program with `Ctrl-C` when matplotlib figure is in focus."""
        if event.key == "ctrl+c":
            plt.close()
            self._stop = True

    def save(self, filename):
        writer = animation.FFMpegWriter(fps=self.fps, codec="h264")
        self.anim.save(filename, writer=writer)


def run_one_impl(args):
    _run_once(args, plot_solution=True)


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


class ForwardEulerSolver:
    def __init__(self, problem):
        self.y = np.copy(problem.y0)
        self.ydot = np.empty_like(problem.y0)
        self.t = 0.0
        self.problem = problem

    def integrate(self, t):
        self.problem.compute_rhs(t, self.y, self.ydot)
        dt = t - self.t
        self.y += dt * self.ydot
        self.t = t


def _run_once(args, N=255, plot_solution=True) -> float:
    if isinstance(args, argparse.Namespace):
        impl = args.impl
    else:
        impl = args
    print("================================================================")
    print(
        "Solving Gray--Scott system with boundary conditions "
        f"with time integration {impl}, N = {N}"
    )
    begin_time = time.time()
    problem = GrayScottProblem(N=N, seed=42)
    times = np.arange(problem.t0, problem.tfinal + problem.dt, step=problem.dt)
    if impl == "native_sundials_cvode":
        print(f"Use native impl {impl}")
        # y = np.copy(problem.y0)
        # problem.plot_2D_solution(y)
        # ydot = np.empty_like(y)
        s = ode(
            "cvode",
            problem.compute_rhs,
            old_api=False,
            lmm_type="ADAMS",
            nonlinsolver="fixedpoint",
        )
        s.init_step(problem.t0, problem.y0)

        for t in times[1:]:
            output = s.step(t)
        y = output.values.y
    elif impl == "sundials_cvode":
        s = IVP(impl)
        s.set_initial_value(problem.y0, problem.t0)
        s.set_rhs_fn(problem.compute_rhs)

        soln = [problem.y0]
        for t in times[1:]:
            s.integrate(t)
        y = s.y
    elif impl == "forward_euler":
        print(f"Use native impl {impl}")
        times = np.arange(problem.t0, problem.tfinal + problem.dt, step=problem.dt)
        s = ForwardEulerSolver(problem)
        if args.anim_play or args.anim_save:
            anim = AnimatedSolution(problem, s, times)
            if args.anim_play:
                plt.show()
            else:
                anim.save("animation.mp4")
        else:
            for t in times:
                s.integrate(t)
            y = s.y
            problem.plot_2D_solution(y)
            plt.tight_layout(pad=0.1)
            filename = RESULT_SOLUTION_FILENAME_TPL.format("forward_euler")
            plt.savefig(filename)
        # extra_options = {"lmm_type": "ADAMS", "old_api": False}
        # ode_solver = ode("CVODE", problem.compute_rhs, **extra_options)
        # output = ode_solver.solve(times, problem.y0)
        # y = output.values.y[-1]

    soln = [problem.y0]
    soln.append(y)
    end_time = time.time()
    elapsed_time = end_time - begin_time
    print("Finished")

    if plot_solution:
        problem.plot_2D_solution(soln[-1])
        plt.tight_layout(pad=0.1)
        filename = RESULT_SOLUTION_FILENAME_TPL.format(impl)
        plt.savefig(filename)

    return elapsed_time


if __name__ == "__main__":
    try:
        args = _parse_args()
        args.func(args)
    except KeyboardInterrupt:
        print("Received Ctrl-C. Finishing...")
