import zlib

import numpy as np
import pytest

import loader


def pytest_report_header(config):
    return f"robolings target: {loader.TARGET_DIR}"


@pytest.fixture(autouse=True)
def fresh_rng(request):
    """Give every test its own random numbers.

    The test modules draw from a module-level ``RNG``. Rebinding it before
    each test, seeded from the test's own name, means a test sees the same
    inputs whether it runs alone (``python robolings.py run slerp``) or
    after every other test in its file (``python robolings.py``).
    """
    if not hasattr(request.module, "RNG"):
        return
    owner = request.cls.__name__ if request.cls is not None else ""
    seed = zlib.crc32(f"{owner}::{request.node.name}".encode())
    request.module.RNG = np.random.default_rng(seed)
