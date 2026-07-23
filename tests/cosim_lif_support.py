# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_cosim_lif.py

from __future__ import annotations

"""LIF Q4.12 and Q16.16 precision contracts."""
import subprocess
import sys
import pytest
from sc_neurocore.compiler.equation_compiler import Q88
from tests.cosim_support import (
    HAS_IVERILOG,
    _lif_schema_precision_values,
    _python_spike_count,
    _verilog_spike_count,
    _verilog_spike_count_q1616,
    _verilog_spike_count_q412,
)
_N_STEPS = 200
_INPUT_CURRENT = 50.0

__all__ = ['subprocess', 'sys', 'pytest', 'Q88', 'HAS_IVERILOG', '_lif_schema_precision_values', '_python_spike_count', '_verilog_spike_count', '_verilog_spike_count_q1616', '_verilog_spike_count_q412', '_N_STEPS', '_INPUT_CURRENT']
