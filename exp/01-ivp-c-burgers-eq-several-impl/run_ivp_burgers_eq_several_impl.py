"""Plot solutions of Burgers' equation obtained with 3 implementations."""

import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np

IMPL_LIST = ["sundials_cvode", "scipy_ode_dopri5", "jl_diffeq"]

OUTDIR = "exp/01-ivp-c-burgers-eq-several-impl/_output"
STYLES = ["-", "--", ":"]
DATA_FILENAME_TPL = "ivp_c_burgers_eq_{:s}.txt"
RESULT_FIG_FILENAME = os.path.join(OUTDIR, "ivp_c_burgers_eq.pdf")


def main():
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
    os.makedirs(OUTDIR, exist_ok=True)
    prog = "build/examples/call_ivp_from_c_burgers_eq"
    for impl in IMPL_LIST:
        outfile = os.path.join(OUTDIR, DATA_FILENAME_TPL.format(impl))
        subprocess.run([prog, impl, outfile])


def plot():
    plt.figure()

    for i, impl in enumerate(IMPL_LIST):
        fn = os.path.join(OUTDIR, DATA_FILENAME_TPL.format(impl))
        data = np.loadtxt(fn)
        x, soln = data[:, 0], data[:, 1]
        plt.plot(x, soln, STYLES[i], label=impl)

    plt.xlabel("$x$")
    plt.ylabel("Solution")
    plt.legend(loc="upper right")
    plt.tight_layout(pad=0.1)
    plt.savefig(RESULT_FIG_FILENAME)
    plt.show()


if __name__ == "__main__":
    main()
