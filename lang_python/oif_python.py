#!/usr/bin/env python3

import ctypes


def execute(lib: ctypes.CDLL):
    lib.oif_init_connector("julia")
    lib.oif_init_lang()
    lib.oif_eval_expression("print('python rocks')")
    lib.oif_deinit_lang()


def load():
    return ctypes.CDLL("liboif_connector.so")


if __name__ == "__main__":
    execute(load())
