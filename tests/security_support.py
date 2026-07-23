# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_security.py

from __future__ import annotations

"""
Comprehensive test suite for SC-NeuroCore Security Modules.

Tests:
- AsimovGovernor (ethics.py) - Three Laws of Robotics enforcement
- DigitalImmuneSystem (immune.py) - Anomaly detection
- WatermarkInjector (watermark.py) - Model fingerprinting
- ZKPVerifier (zkp.py) - Zero-knowledge proofs for spike validity
"""
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

    def __init__(self, n_neurons, n_inputs):
        self.weights = np.random.rand(n_neurons, n_inputs)
        self._refresh_called = False

    def _refresh_packed_weights(self):
        self._refresh_called = True
class MockLayerNoWeights:
    """Mock layer without weights attribute."""

    pass
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

__all__ = ['pytest', 'np', 'sys', 'os', 'AsimovGovernor', 'ActionRequest', 'DigitalImmuneSystem', 'WatermarkInjector', 'ZKPVerifier', 'MockLayer', 'MockLayerNoWeights']
