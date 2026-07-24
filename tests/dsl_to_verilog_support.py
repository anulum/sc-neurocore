# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_dsl_to_verilog.py

from __future__ import annotations

"""End-to-end tests: schema → UniversalNeuron → Verilog → Icarus Verilog.

These tests validate the complete pipeline from TOML/JSON model schemas
through the equation compiler to synthesizable Verilog RTL, and then
compile + simulate with Icarus Verilog to verify functional correctness.
"""
import subprocess
import tempfile
from pathlib import Path
import pytest
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from sc_neurocore.compiler.equation_compiler import (
    generate_testbench,
)
import shutil

HAS_IVERILOG = shutil.which("iverilog") is not None
_SIMPLE_MODELS = [
    "lif",
    "lapicque",
    "izhikevich",
    "quadratic_if",
    "resonate_fire",
    "fitzhugh_nagumo",
]
_TRANSCENDENTAL_MODELS = ["adex", "hindmarsh_rose"]

__all__ = [
    "subprocess",
    "tempfile",
    "Path",
    "pytest",
    "UniversalNeuron",
    "generate_testbench",
    "shutil",
    "HAS_IVERILOG",
    "_SIMPLE_MODELS",
    "_TRANSCENDENTAL_MODELS",
]
