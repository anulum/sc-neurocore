# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_snn_modules.py

from __future__ import annotations

"""Tests for differentiable SNN modules."""
import pytest
torch = pytest.importorskip("torch")
from sc_neurocore.training.snn_modules import (
    ALIFCell,
    AdExCell,
    AlphaCell,
    ConvSpikingNet,
    ExpIFCell,
    IFCell,
    LapicqueCell,
    LIFCell,
    RecurrentLIFCell,
    SCWeightNoiseModel,
    SecondOrderLIFCell,
    SpikingNet,
    SynapticCell,
)
from sc_neurocore.training.surrogate import superspike

__all__ = ['pytest', 'torch', 'ALIFCell', 'AdExCell', 'AlphaCell', 'ConvSpikingNet', 'ExpIFCell', 'IFCell', 'LapicqueCell', 'LIFCell', 'RecurrentLIFCell', 'SCWeightNoiseModel', 'SecondOrderLIFCell', 'SpikingNet', 'SynapticCell', 'superspike', '__all__']
