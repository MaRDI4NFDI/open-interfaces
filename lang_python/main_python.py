#!/usr/bin/env python3

import ctypes
import os


def execute(lib: ctypes.CDLL):
    lib.oif_connector_init("julia")
    lib.oif_connector_eval_expression("print('python rocks')")
    lib.oif_connector_deinit()


def load():
    soname = "liboif_connector.so"
    try:
        return ctypes.CDLL(soname)
    except OSError as e:
        print(f"could not load {soname}")
        print(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH')}")
        raise e


if __name__ == "__main__":
    execute(load())
