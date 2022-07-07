import glob
import logging

import pytest
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def oif_env():
    try:
        load_dotenv(dotenv_path="oif_pytest.env", override=True)
    except FileNotFoundError as e:
        for fn in glob.glob("cmake-*/oif_pytest.env"):
            try:
                load_dotenv(dotenv_path=fn, override=True)
                logging.warning(f"loaded environment from fallback {fn}")
                return
            except FileNotFoundError:
                continue
        raise e
