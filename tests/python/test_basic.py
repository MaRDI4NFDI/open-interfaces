import ctypes
import sys
from typing import Tuple

import pytest


def test_print(oif_lib: Tuple[ctypes.CDLL, str], capfd) -> None:
    lib, lang = oif_lib
    err = lib.oif_connector_init(lang.encode())
    assert err == 0
    expression = "print(6*7)".encode()

    ret: int = lib.oif_connector_eval_expression(expression)
    out, err = capfd.readouterr()
    if ret != 0:
        assert ret == 4  # not implemented
    else:
        assert "42" in out or "42" in err


def test_fail_eval(oif_lib: Tuple[ctypes.CDLL, str]) -> None:
    lib, lang = oif_lib
    err = lib.oif_connector_init(lang.encode())
    assert err == 0

    expression = 'blbla("foobar")'.encode()
    ret: int = lib.oif_connector_eval_expression(expression)
    assert ret != 0


def runmodule(filename):
    sys.exit(pytest.main(sys.argv[1:] + [filename]))


if __name__ == "__main__":
    runmodule(filename=__file__)
