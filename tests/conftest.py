# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared test configuration and fixtures for SC-NeuroCore

"""
Shared test configuration and fixtures for SC-NeuroCore.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def restore_repo_cwd() -> Iterator[None]:
    """Keep process-wide CWD changes from leaking across tests."""
    os.chdir(_REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(_REPO_ROOT)


@pytest.fixture(autouse=True)
def seed_random() -> Iterator[None]:
    """Seed numpy RNG before every test for deterministic results."""
    np.random.seed(42)
    yield
