import argparse

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("filename", type=str, help="CSV file with memory usage records")
    p.add_argument("-s", "--save", action="store_true", help="Save figure to disk")
    return p.parse_args()


def main():
    args = parse_args()

    data = np.loadtxt(args.filename, delimiter=",", skiprows=1)
    bytes_in_mib = 1024**2

    plt.figure()
    plt.plot(data[:, 0] / bytes_in_mib, "-", label="RSS")
    plt.plot(data[:, 1] / bytes_in_mib, "--", label="VMS")
    plt.legend(loc="best")
    plt.xlabel("Run time, index")
    plt.ylabel("Memory, MiB")
    plt.tight_layout(pad=0.1)
    plt.show()

    if args.save:
        figname = args.filename.replace(".csv", ".pdf")
        plt.savefig(figname)


if __name__ == "__main__":
    main()
