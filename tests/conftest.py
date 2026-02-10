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
