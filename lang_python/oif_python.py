import ctypes
import os


def execute(oif: ctypes.CDLL):
    oif.oif_init_lang()
    oif.oif_eval_expression("print('python rocks')")
    oif.oif_deinit_lang()


def load():
    return ctypes.CDLL("liboif_connector.so")


if __name__ == "__main__":
    execute(load())
