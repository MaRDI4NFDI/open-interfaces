"""Solve van der Pol' equation obtained with different implementations."""

import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np

IMPL_LIST = ["scipy_ode", "jl_diffeq"]

OUTDIR = "exp/03-ivp-py-van-der-pol-eq-several-impl/_output"
STYLES = ["-", "--", ":"]
DATA_FILENAME_TPL = "ivp_py_vdp_eq_{:s}.txt"
RESULT_FIG_FILENAME = os.path.join(OUTDIR, "ivp_py_vdp_eq.pdf")


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    if not data_are_present():
        compute()
    else:
        plot()


def data_are_present():
    for impl in IMPL_LIST:
        fn = os.path.join(OUTDIR, DATA_FILENAME_TPL.format(impl))
        if not os.path.isfile(fn):
            print("'{:s}' is not a file".format(fn))
            return False
    return True


def compute():
    prog = "examples/call_ivp_from_python_vdp.py"
    for impl in IMPL_LIST:
        subprocess.run(["python", prog, impl, "--outdir", OUTDIR])


def plot():
    plt.figure()

    for i, impl in enumerate(IMPL_LIST):
        fn = os.path.join(OUTDIR, DATA_FILENAME_TPL.format(impl))
        data = np.loadtxt(fn)
        t, y1 = data[:, 0], data[:, 1]
        plt.plot(t, y1, STYLES[i], label=impl)

    plt.xlabel("Time")
    plt.ylabel("Solution")
    plt.legend(loc="upper right")
    plt.tight_layout(pad=0.1)
    plt.savefig(RESULT_FIG_FILENAME, transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
