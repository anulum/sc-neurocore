# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_security.py

"""Mock layers for the security-module tests, after the source path is put first on sys.path.

Importing this module inserts the repository's src directory at the front of sys.path, then imports the
security classes; importers that take those classes from here run that insertion first.
"""

from __future__ import annotations

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sc_neurocore.security.ethics import AsimovGovernor, ActionRequest
from sc_neurocore.security.immune import DigitalImmuneSystem
from sc_neurocore.security.watermark import WatermarkInjector
from sc_neurocore.security.zkp import ZKPVerifier


class MockLayer:
    """Mock layer with weights for watermark testing."""

    def __init__(self, n_neurons: int, n_inputs: int) -> None:
        """Create random weights of shape (n_neurons, n_inputs)."""
        self.weights = np.random.rand(n_neurons, n_inputs)
        self._refresh_called = False

    def _refresh_packed_weights(self) -> None:
        """Record that the injector asked for a weight refresh."""
        self._refresh_called = True


class MockLayerNoWeights:
    """Mock layer without weights attribute."""

    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

__all__ = [
    "pytest",
    "np",
    "sys",
    "os",
    "AsimovGovernor",
    "ActionRequest",
    "DigitalImmuneSystem",
    "WatermarkInjector",
    "ZKPVerifier",
    "MockLayer",
    "MockLayerNoWeights",
]
