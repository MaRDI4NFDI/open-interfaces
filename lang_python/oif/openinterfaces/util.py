"""Utility function and classes for Python user-facing codes."""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import psutil
from scipy import sparse as sp


class UsedMemoryMonitor:
    """Memory monitoring class that dumps memory usage to a CSV file."""

    def __init__(self, csv_filename="memory_usage.csv"):
        self.csv_filename = csv_filename
        self.headers = ["rss", "vms", "shared", "text", "lib", "data", "dirty"]
        self.data = []
        if os.path.isfile(csv_filename):
            print(f"[UsedMemoryMonitor] Reading existing file '{csv_filename}'")
        else:
            print(f"[UsedMemoryMonitor] Creating new file '{csv_filename}'")
            self.fh = open(self.csv_filename, "w")
            self.writer = csv.writer(self.fh)
            self.writer.writerow(self.headers)

    def record(self):
        # Get the current process
        current_process = psutil.Process(os.getpid())

        # Get the memory info
        memory_info = current_process.memory_info()

        # Prepare data for CSV
        row = [
            memory_info.rss,
            memory_info.vms,
            memory_info.shared,
            memory_info.text,
            memory_info.lib,
            memory_info.data,
            memory_info.dirty,
        ]

        # Write to CSV
        self.writer.writerow(row)

    def plot(self):
        data = np.loadtxt(self.csv_filename, delimiter=",", skiprows=1)

        plt.figure()
        plt.plot(data[:, 0], "-", label="RSS")
        plt.plot(data[:, 1], "--", label="VMS")
        plt.ylabel("Memory, MiB")
        plt.legend(loc="best")
        plt.show()

    def __del__(self):
        self.fh.close()


class Laplacian2DApproximator:
    def __init__(self, N, dx, dy):
        self.N = N
        self.dx = dx
        self.dy = dy

        self.A = self.laplacian2DMatrix()

    @property
    def matrix(self):
        return self.A

    def laplacian2DMatrix(self):
        N = self.N
        dx, dy = self.dx, self.dy
        diag = np.ones([N * N])
        Id = sp.eye(N)
        mat_x = sp.spdiags([diag, -2 * diag, diag], [-1, 0, 1], N, N) / dx**2
        mat_y = sp.spdiags([diag, -2 * diag, diag], [-1, 0, 1], N, N) / dy**2
        return sp.kron(Id, mat_x, format="csr") + sp.kron(mat_y, Id)

    def laplacian_periodic(self, U):
        """Returns 5-point 2D Laplacian approximation of quantity U."""
        N = self.N
        dx, dy = self.dx, self.dy
        assert N == len(U)
        AU = self.A.dot(U.reshape((-1,)))
        AU_2D = np.reshape(AU, (N, N))

        # Top boundary
        AU_2D[0, 1:-1] = (U[0, 0:-2] - 2 * U[0, 1:-1] + U[0, 2:]) / dx**2 + (
            U[-2, 1:-1] - 2 * U[0, 1:-1] + U[1, 1:-1]
        ) / dy**2
        # Bottom boundary
        AU_2D[-1, 1:-1] = AU_2D[0, 1:-1]

        # Left boundary
        AU_2D[1:-1, 0] = (U[1:-1, -2] - 2 * U[1:-1, 0] + U[1:-1, 1]) / dx**2 + (
            U[0:-2, 0] - 2 * U[1:-1, 0] + U[2:, 0]
        ) / dy**2
        # Right boundary.
        AU_2D[1:-1, -1] = AU_2D[1:-1, 0]

        AU_2D[0, 0] = (U[0, -2] - 2 * U[0, 0] + U[0, 1]) / dx**2 + (
            U[-1, 0] - 2 * U[0, 0] + U[1, 0]
        ) / dy**2
        AU_2D[0, -1] = AU_2D[0, 0]
        AU_2D[-1, 0] = AU_2D[0, 0]
        AU_2D[-1, -1] = AU_2D[0, 0]

        return AU_2D
