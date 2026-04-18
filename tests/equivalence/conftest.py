# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared fixtures for v2-vs-v3 equivalence testing

"""Shared fixtures for v2-vs-v3 equivalence testing."""

import numpy as np
import pytest


@pytest.fixture
def deterministic_rng():
    """Return a seeded NumPy RNG for reproducible tests."""
    return np.random.RandomState(42)


@pytest.fixture
def sample_bitstream():
    """A known 1024-bit bitstream for testing."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 2, 1024).astype(np.uint8)
