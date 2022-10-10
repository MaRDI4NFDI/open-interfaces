#!/usr/bin/env python3

import ctypes
from typing import Tuple

import numpy as np

DOUBLE_STAR = ctypes.POINTER(ctypes.c_double)


def test_solve(oif_lib: Tuple[ctypes.CDLL, str]):
    lib, lang = oif_lib
    err = lib.oif_connector_init(lang.encode())
    assert err == 0

    N = 2
    A = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=float)
    b = np.array([1.0, 1.0], dtype=float)
    x = np.array([-10.0, 0.0], dtype=float)
    ret = lib.oif_connector_solve(
        N,
        A.ctypes.data_as(DOUBLE_STAR),
        b.ctypes.data_as(DOUBLE_STAR),
        x.ctypes.data_as(DOUBLE_STAR),
    )

    if lang in ["c", "cpp"]:
        assert ret == 0
    else:
        if ret != 0:
            assert ret == 4  # not implemented
        return
    print(f"solution from {lang}: {x}")
    assert np.allclose(np.dot(A, x), b)
