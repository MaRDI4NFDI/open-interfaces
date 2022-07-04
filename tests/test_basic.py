import os
import subprocess
import sys
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).resolve().absolute().parent
REPO_ROOT = THIS_DIR.parent
SOURCE_EXTENSIONS = [".c", ".cpp", ".cc"]


def drivers() -> Path:
    for lang_dir in REPO_ROOT.glob("lang_*"):
        lang = lang_dir.stem.replace("lang_", "")
        for driver in lang_dir.glob(f"main_{lang}*"):
            if driver.suffix in SOURCE_EXTENSIONS:
                build_dir = os.environ.get("M2_BUILD_DIR", None)
                if build_dir:
                    yield Path(build_dir).resolve() / f"lang_{lang}" / driver.stem
                else:
                    # assume binary in path
                    yield driver.stem
                continue
            if driver.exists() and driver.is_file():
                yield driver


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
    "driver", drivers(), ids=(f if isinstance(f, str) else f.name for f in drivers())
)
@pytest.mark.parametrize("lang", languages())
def test_print(driver, lang) -> None:
    expression = "print(6*7)"
    print(f"Execute {driver} {lang} {expression}")
    out = subprocess.check_output([driver, lang, expression]).decode()
    # string eval not implemented
    if lang not in ["c", "cpp"]:
        assert "42" in out


def runmodule(filename):
    sys.exit(pytest.main(sys.argv[1:] + [filename]))


if __name__ == "__main__":
    runmodule(filename=__file__)
