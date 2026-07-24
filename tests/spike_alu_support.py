# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_spike_alu.py

from __future__ import annotations

"""Tests for SpikeGate, SpikeRegister, SpikeALU, spike_sort."""
import numpy as np
import pytest
from sc_neurocore.symbolic.spike_logic import (
    SpikeGate,
    SpikeRegister,
    SpikeALU,
    spike_sort,
)

__all__ = ["np", "pytest", "SpikeGate", "SpikeRegister", "SpikeALU", "spike_sort"]
