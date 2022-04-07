#!/usr/bin/env python3

import ctypes


def execute(lib: ctypes.CDLL):
    lib.oif_connector_init("julia")
    lib.oif_connector_eval_expression("print('python rocks')")
    lib.oif_connector_deinit()


def load():
    return ctypes.CDLL("liboif_connector.so")


if __name__ == "__main__":
    execute(load())
