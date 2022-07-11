import ctypes
import itertools
import logging
import os
from pathlib import Path
from typing import Tuple, Callable

import pytest
from dotenv import load_dotenv

THIS_DIR = Path(__file__).resolve().absolute().parent
REPO_ROOT = THIS_DIR.parent.parent
SOURCE_EXTENSIONS = [".c", ".cpp", ".cc"]
DRIVER_TYPE = Tuple[str, Callable]
DOUBLE_STAR = ctypes.POINTER(ctypes.c_double)


@pytest.fixture(scope="session", autouse=True)
def oif_env() -> bool:
    for fn in itertools.chain(
        [Path("oif_pytest.env")], REPO_ROOT.glob("cmake-*/oif_pytest.env")
    ):
        try:
            if not (fn.exists() and fn.is_file()):
                continue
            if load_dotenv(dotenv_path=fn, override=True):
                logging.warning(f"loaded environment from {fn}")
                return True
        except FileNotFoundError:
            continue
    return False


@pytest.fixture(autouse=True)
def change_test_dir(request, oif_env):
    os.chdir(os.environ.get("M2_BINARY_DIR"))
    yield
    os.chdir(request.config.invocation_dir)


def languages() -> str:
    for lang_dir in REPO_ROOT.glob("lang_*"):
        lang = lang_dir.stem.replace("lang_", "")
        for ext in SOURCE_EXTENSIONS:
            source = lang_dir / f"oif_{lang}{ext}"
            if source.exists() and source.is_file():
                yield lang
                break
        else:
            glob = list(lang_dir.glob("oif_*"))
            pytest.fail(f"No source file for {lang_dir}:{glob}")


@pytest.fixture(params=languages())
def oif_lib(request, oif_env) -> ctypes.CDLL:
    assert oif_env
    lang = request.param
    soname = "./oif_connector/liboif_connector.so"
    # With PyDLL we do not release the GIL when calling into lib functions
    dso_type = ctypes.PyDLL if lang == "python" else ctypes.CDLL
    try:
        lib = dso_type(soname)
    except OSError as e:
        print(f"could not load {soname}")
        ld = os.environ.get("LD_LIBRARY_PATH")
        print(f"LD_LIBRARY_PATH={ld}")
        raise e
    lib.oif_connector_eval_expression.argtypes = [ctypes.c_char_p]
    lib.oif_connector_init.argtypes = [ctypes.c_char_p]
    lib.oif_connector_solve.argtypes = [
        ctypes.c_int,
        DOUBLE_STAR,
        DOUBLE_STAR,
        DOUBLE_STAR,
    ]
    yield lib, lang
    lib.oif_connector_deinit()
