"""Utility function and classes for Python user-facing codes."""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import psutil


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
