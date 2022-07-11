import ctypes
from typing import Tuple


def test_init(oif_lib: Tuple[ctypes.CDLL, str]) -> None:
    lib, lang = oif_lib
    err = lib.oif_connector_init(lang.encode())
    assert err == 0
