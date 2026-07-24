# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_neural_decoders.py

from __future__ import annotations

"""Tests for neural_decoders: POYO+, POSSM, NDT3, CEBRA.

Multi-angle, publication-property tests — not just happy path.
"""
import numpy as np
import pytest
from sc_neurocore.analysis.spike_stats.neural_decoders import (
    CEBRAEncoder,
    NDT3Decoder,
    POSSMDecoder,
    POYODecoder,
    scaled_dot_product_attention,
    sinusoidal_position_encode,
    tokenise_spikes,
)

__all__ = [
    "np",
    "pytest",
    "CEBRAEncoder",
    "NDT3Decoder",
    "POSSMDecoder",
    "POYODecoder",
    "scaled_dot_product_attention",
    "sinusoidal_position_encode",
    "tokenise_spikes",
]
