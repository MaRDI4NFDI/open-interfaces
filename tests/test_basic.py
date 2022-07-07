import functools
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Tuple

import pytest

THIS_DIR = Path(__file__).resolve().absolute().parent
REPO_ROOT = THIS_DIR.parent
SOURCE_EXTENSIONS = [".c", ".cpp", ".cc"]
DRIVER_TYPE = Tuple[str, Callable]


def drivers() -> DRIVER_TYPE:
    for lang_dir in REPO_ROOT.glob("lang_*"):
        lang = lang_dir.stem.replace("lang_", "")
        for driver in lang_dir.glob(f"main_{lang}*"):
            if driver.suffix in SOURCE_EXTENSIONS:
                # env eval needs to be deferred to runtime
                def _callable(_driver, _lang, build_dir):
                    binary = Path(build_dir).resolve() / f"lang_{_lang}" / _driver.stem
                    if binary.exists() and binary.is_file():
                        return binary
                    else:
                        # assume binary in path
                        return _driver.stem

                yield driver.stem, functools.partial(_callable, driver, lang)
                continue
            elif driver.exists() and driver.is_file():

                def _callable(_driver, build_dir):
                    return _driver

                yield driver.name, functools.partial(_callable, driver)
                continue
            assert False, driver


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


@pytest.mark.parametrize(
    "driver", [d for _, d in drivers()], ids=[f for f, _ in drivers()]
)
@pytest.mark.parametrize("lang", languages())
def test_print(driver: Callable, lang: str) -> None:
    expression = "print(6*7)"
    driver = driver(build_dir=os.environ["M2_BINARY_DIR"])
    print(f"Execute {driver} {lang} {expression}")
    out = subprocess.check_output([driver, lang, expression]).decode()
    # string eval not implemented
    if lang not in ["c", "cpp"]:
        assert "42" in out


def runmodule(filename):
    sys.exit(pytest.main(sys.argv[1:] + [filename]))


if __name__ == "__main__":
    runmodule(filename=__file__)
