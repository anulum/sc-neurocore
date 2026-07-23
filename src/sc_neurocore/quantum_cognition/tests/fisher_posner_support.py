# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_fisher_posner.py

from __future__ import annotations

"""Inline tests for HybridFisherPosnerLIF."""
import pytest
from sc_neurocore.quantum_cognition.fisher_posner import (
    HybridFisherPosnerLIF,
    HybridFisherPosnerLIFNeuron,
)
from sc_neurocore.quantum_cognition.spin_pool import SpinPoolMPS
PoolAndNeuron = tuple[SpinPoolMPS, HybridFisherPosnerLIF]
@pytest.fixture
def pool_and_neuron() -> PoolAndNeuron:
    pool = SpinPoolMPS(n_sites=8)
    neuron = HybridFisherPosnerLIF(neuron_id=0, spin_pool=pool)
    return pool, neuron

__all__ = ['pytest', 'HybridFisherPosnerLIF', 'HybridFisherPosnerLIFNeuron', 'SpinPoolMPS', 'PoolAndNeuron', 'pool_and_neuron']
