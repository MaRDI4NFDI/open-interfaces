"""Auxiliary script to plot PDE solutions with one spatial variable."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("data_filename", type=str, help="Solution filename")
p.add_argument("image_filename", type=str, help="Filename for plotted image")
args = p.parse_args()

data_filename = args.data_filename
image_filename = args.image_filename

data = np.loadtxt(data_filename)
x, soln = data[:, 0], data[:, 1]

plt.figure()
plt.plot(x, soln, "-")
plt.xlabel("$x$")
plt.ylabel("Solution")
plt.tight_layout(pad=0.1)
plt.savefig(image_filename)
plt.show()
