# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_spikeinterface_adapter.py

from __future__ import annotations

"""Tests for spike train → bitstream/population conversion."""
import numpy as np
from sc_neurocore.adapters.spikeinterface import (
    firing_rates_to_sc_probs,
    spike_trains_to_bitstreams,
    spike_trains_to_population_input,
)

__all__ = ['np', 'firing_rates_to_sc_probs', 'spike_trains_to_bitstreams', 'spike_trains_to_population_input']
