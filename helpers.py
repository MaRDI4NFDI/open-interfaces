"""Auxiliary module to simplify plotting."""

import os

import matplotlib.pyplot as plt

# Figure size for a single-plot figure that takes 50 % of text width.
FIGSIZE_NORMAL = (3.0, 2)
# Figure size for a single-plot figure that takes about 75 % of text width.
FIGSIZE_LARGER = (4.5, 3)
# Figure size for a two-subplots figure.
FIGSIZE_TWO_SUBPLOTS_TWO_ROWS = (3.0, 4.0)
# Figure size for a figure with two subplots in one row.
FIGSIZE_TWO_SUBPLOTS_ONE_ROW = (6.0, 2)
# Figure size for a figure with two subplots in two rows.
FIGSIZE_WIDE_TWO_SUBPLOTS_TWO_ROWS = (4.5, 4)


def savefig(filename, dirname="", **kwargs):
    """Save figure if the environment variable SAVE_FIGURES is set."""
    cur_fig = plt.gcf()

    if dirname or "SAVE_FIGURES" in os.environ:
        if os.path.isdir(dirname):
            filename = os.path.join(dirname, filename)
            cur_fig.savefig(filename, **kwargs)
        else:
            raise RuntimeError("Directory `%s` does not exist" % dirname)
    else:
        plt.show()
