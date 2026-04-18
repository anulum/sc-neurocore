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

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def seed_random():
    """Seed numpy RNG before every test for deterministic results."""
    np.random.seed(42)
    yield
