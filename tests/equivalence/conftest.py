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
