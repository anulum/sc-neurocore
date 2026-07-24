# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_stochastic_core_properties.py

from __future__ import annotations

"""Property-based tests for stochastic core contracts using Hypothesis."""

import numpy as np

from hypothesis import given, settings

from hypothesis import strategies as st

from sc_neurocore import (
    BitstreamEncoder,
    generate_bernoulli_bitstream,
    bitstream_to_probability,
    StochasticLIFNeuron,
    FixedPointLIFNeuron,
    FixedPointLFSR,
    FixedPointBitstreamEncoder,
    SCDenseLayer,
    VectorizedSCLayer,
    RNG,
    BitstreamSpikeRecorder,
    HomeostaticLIFNeuron,
    SCIzhikevichNeuron,
    StochasticSTDPSynapse,
    RewardModulatedSTDPSynapse,
)

from sc_neurocore.constants import (
    LIF_V_REST,
    LIF_V_THRESHOLD,
    FP_DATA_WIDTH,
)


__all__ = ['np', 'given', 'settings', 'st', 'BitstreamEncoder', 'generate_bernoulli_bitstream', 'bitstream_to_probability', 'StochasticLIFNeuron', 'FixedPointLIFNeuron', 'FixedPointLFSR', 'FixedPointBitstreamEncoder', 'SCDenseLayer', 'VectorizedSCLayer', 'RNG', 'BitstreamSpikeRecorder', 'HomeostaticLIFNeuron', 'SCIzhikevichNeuron', 'StochasticSTDPSynapse', 'RewardModulatedSTDPSynapse', 'LIF_V_REST', 'LIF_V_THRESHOLD', 'FP_DATA_WIDTH']
