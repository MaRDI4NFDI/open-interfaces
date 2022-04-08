import itertools
import subprocess
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).resolve().absolute().parent
REPO_ROOT = THIS_DIR.parent


def drivers() -> Path:
    for lang_dir in REPO_ROOT.glob("lang_*"):
        lang = lang_dir.stem.replace("lang_", "")
        for driver in lang_dir.glob(f"main_{lang}*"):
            if driver.suffix in [".c", ".cpp", ".cc"]:
                # TODO assumes built binary in path
                yield driver.stem
                continue
            if driver.exists() and driver.is_file():
                yield driver


def languages() -> str:
    for lang_dir in REPO_ROOT.glob("lang_*"):
        lang = lang_dir.stem.replace("lang_", "")
        c_source = lang_dir / f"oif_{lang}.c"
        assert c_source.exists() and c_source.is_file()
        yield lang


@pytest.mark.parametrize(
    "driver", drivers(), ids=(f if isinstance(f, str) else f.name for f in drivers())
)
@pytest.mark.parametrize("lang", languages())
def test_print(driver, lang) -> None:
    if lang == "c":
        pytest.xfail("string eval not implemented")
    expression = "print(6*7)"
    print(f"Execute {driver} {lang} {expression}")
    out = subprocess.check_output([driver, lang, expression]).decode()
    assert "42" in out
