# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_synapses_stdp.py

from __future__ import annotations

"""
Tests for StochasticSTDPSynapse and RewardModulatedSTDPSynapse.
Covers the untested code paths in stochastic_stdp.py and r_stdp.py.
"""
import pytest
import numpy as np
from sc_neurocore.synapses.stochastic_stdp import StochasticSTDPSynapse
from sc_neurocore.synapses.r_stdp import RewardModulatedSTDPSynapse

__all__ = ['pytest', 'np', 'StochasticSTDPSynapse', 'RewardModulatedSTDPSynapse']
