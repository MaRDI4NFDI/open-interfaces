#!/usr/bin/env python3

import ctypes
import os
import sys

import numpy as np


DOUBLE_STAR = ctypes.POINTER(ctypes.c_double)


def load(lang) -> ctypes.CDLL:
    soname = "./oif_connector/liboif_connector.so"
    # With PyDLL we do not release the GIL when calling into lib functions
    dso_type = ctypes.PyDLL if lang == "python" else ctypes.CDLL
    try:
        lib = dso_type(soname)
    except OSError as e:
        print(f"could not load {soname}")
        print(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH')}")
        raise e
    lib.oif_connector_eval_expression.argtypes = [ctypes.c_char_p]
    lib.oif_connector_init.argtypes = [ctypes.c_char_p]
    lib.oif_connector_solve.argtypes = [
        ctypes.c_int,
        DOUBLE_STAR,
        DOUBLE_STAR,
        DOUBLE_STAR,
    ]
    return lib


if __name__ == "__main__":
    if len(sys.argv) != 3:
        lang = "julia"
        expression = "print(42)"
    else:
        lang = sys.argv[1]
        expression = sys.argv[2]
    lib = load(lang)
    err = lib.oif_connector_init(lang.encode())
    assert err == 0
    err = lib.oif_connector_eval_expression(expression.encode())
    assert err == 0
    N = 2
    A = np.array([2.0, 0.0, 1.0, 0.0], dtype=float)
    b = np.array([1.0, 1.0], dtype=float)
    x = np.array([0.0, 0.0], dtype=float)
    err = lib.oif_connector_solve(
        N,
        A.ctypes.data_as(DOUBLE_STAR),
        b.ctypes.data_as(DOUBLE_STAR),
        x.ctypes.data_as(DOUBLE_STAR),
    )
    print(x)
    lib.oif_connector_deinit()
