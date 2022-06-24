#!/usr/bin/env python3

import ctypes
import os
import sys


def load(lang) -> ctypes.CDLL:
    soname = "liboif_connector.so"
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
    lib.oif_connector_deinit()
