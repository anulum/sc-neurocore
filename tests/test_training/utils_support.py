# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_utils.py

from __future__ import annotations

"""Tests for SpikeMonitor, population_decode, reset_states."""
import pytest
torch = pytest.importorskip("torch")
from sc_neurocore.training.snn_modules import SpikingNet
from sc_neurocore.training.utils import SpikeMonitor, population_decode, reset_states

__all__ = ['pytest', 'torch', 'SpikingNet', 'SpikeMonitor', 'population_decode', 'reset_states']
