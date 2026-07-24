# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_equation_builder_adversarial.py

from __future__ import annotations

"""Adversarial test suite for the equation builder's AST sandbox.

Every test in this module provides a malicious equation string that MUST
be rejected by the sandbox.  If any test fails (i.e. the expression is
accepted), there is a sandbox escape vulnerability.

Categories covered:
- Direct code execution via builtins (__import__, eval, exec)
- Dunder attribute chain escapes (__class__.__bases__, __mro__, etc.)
- Module injection (os, sys, subprocess, importlib)
- AST depth exhaustion
- Obfuscated injection via string manipulation
"""
import pytest
from sc_neurocore.neurons.equation_builder import EquationNeuron, from_equations

__all__ = ["pytest", "EquationNeuron", "from_equations"]
