# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_cosim_hodgkin_huxley.py

from __future__ import annotations

"""Hodgkin-Huxley schema, hand-model, and Q16.16 RTL parity contracts."""
import pytest
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import (
    HAS_IVERILOG,
    _hodgkin_huxley_hand_spike_count,
    _python_spike_count,
    _verilog_compiles,
    _verilog_spike_count_q1616,
)
_TRANSCENDENTAL_COMPILE_MODELS = ["hodgkin_huxley"]

__all__ = ['pytest', 'UniversalNeuron', 'HAS_IVERILOG', '_hodgkin_huxley_hand_spike_count', '_python_spike_count', '_verilog_compiles', '_verilog_spike_count_q1616', '_TRANSCENDENTAL_COMPILE_MODELS']
