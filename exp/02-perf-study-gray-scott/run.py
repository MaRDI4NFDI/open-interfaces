"""Driver script for performance study based on the Gray--Scott problem."""

import os
import pickle
import shutil

import matplotlib.pyplot as plt
import numpy as np

import examples.compare_performance_ivp_cvode_gray_scott as gs

OUTDIR = "exp/02-perf-study-gray-scott/_output"
RESULT_DATA_PICKLE = os.path.join(OUTDIR, "ivp_cvode_gs_data.pickle")
RESULT_PERF_FILENAME = os.path.join(OUTDIR, "ivp_cvode_gs_performance.pdf")


def main():
    if results_are_present():
        analyze()
    else:
        compute()
        move_files_to_outdir()
        print("Computations are done")


def results_are_present() -> bool:
    if os.path.isfile(RESULT_DATA_PICKLE):
        return True
    if os.path.isfile(gs.RESULT_DATA_PICKLE):
        move_files_to_outdir()
        return True

    return False


def analyze():
    with open(RESULT_DATA_PICKLE, "rb") as fh:
        tts_list = pickle.load(fh)

    tts_stats = gs.compute_stats(tts_list)

    gs.print_stats(tts_stats)

    fig, axes = plt.subplots(nrows=2, ncols=1)
    for impl in gs.IMPL_LIST:
        tts_ave = [tts_stats[impl][N]["tts_ave"] for N in gs.RESOLUTIONS]
        tts_std = [tts_stats[impl][N]["tts_std"] for N in gs.RESOLUTIONS]
        axes[0].errorbar(gs.RESOLUTIONS, tts_ave, fmt="-o", yerr=tts_std, label=impl)
    axes[0].legend(loc="best")

    # Plot relative times (normalized by native performance).
    impl = gs.IMPL_LIST[-1]
    assert impl.startswith("native_")
    assert len(gs.IMPL_LIST) == 2
    tts_ave_native = np.array([tts_stats[impl][N]["tts_ave"] for N in gs.RESOLUTIONS])
    tts_std_native = np.array([tts_stats[impl][N]["tts_std"] for N in gs.RESOLUTIONS])

    for impl in gs.IMPL_LIST[:-1]:
        tts_ave = np.array([tts_stats[impl][N]["tts_ave"] for N in gs.RESOLUTIONS])
        tts_std = np.array([tts_stats[impl][N]["tts_std"] for N in gs.RESOLUTIONS])
        tts_std_normalized = np.sqrt(
            np.square(tts_std / tts_ave) + np.square(tts_std_native / tts_ave_native)
        )
        axes[1].errorbar(
            gs.RESOLUTIONS,
            tts_ave / tts_ave_native,
            yerr=tts_std_normalized,
            fmt="-o",
            label=impl,
        )
    plt.tight_layout(pad=0.1)

    plt.savefig(RESULT_PERF_FILENAME)


def compute():
    gs.run_all_impl()


def move_files_to_outdir():
    files = os.listdir("assets")
    for f in files:
        if f.startswith("ivp_cvode_gs"):
            filename = os.path.join("assets", f)
            new_filename = os.path.join(OUTDIR, f)
            shutil.move(filename, new_filename)


if __name__ == "__main__":
    main()
