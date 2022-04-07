#!/usr/bin/env python3

import ctypes
import os


def execute(lib: ctypes.CDLL):
    err = lib.oif_connector_init("julia")
    assert err == 0
    err = lib.oif_connector_eval_expression("print('python rocks')")
    assert err == 0
    err = lib.oif_connector_deinit()
    assert err == 0


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
