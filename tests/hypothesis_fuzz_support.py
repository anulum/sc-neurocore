# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_hypothesis_fuzz.py

from __future__ import annotations

"""Property-based fuzz tests using Hypothesis.

Targets:
1. EquationNeuron — ensure random equation strings never escape the sandbox
2. sanitize_ident — ensure no injection-capable identifier passes through
3. VerilogGenerator — ensure random module names + layer configs don't crash
4. UniversalNeuron — ensure random schema dicts don't produce unexpected state

Strategy: generate random but structurally-valid inputs and assert invariants
(no crashes, no code execution, no NaN propagation).
"""
import math
import string
import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from sc_neurocore.hdl_gen._ident import sanitize_ident
from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
_OPERATORS = st.sampled_from(["+", "-", "*", "/", "**"])
_NUMBERS = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
_SAFE_VARS = st.sampled_from(["v", "w", "u", "x", "y", "z", "I", "theta"])
_SAFE_FUNCS = st.sampled_from(["sin", "cos", "exp", "tanh", "sqrt", "abs"])
_SIMPLE_EXPR = st.builds(
    lambda var, op, num: f"{var} {op} {num}",
    _SAFE_VARS,
    _OPERATORS,
    _NUMBERS,
)
_FUNC_EXPR = st.builds(
    lambda fn, var, num: f"{fn}({var} + {num})",
    _SAFE_FUNCS,
    _SAFE_VARS,
    _NUMBERS,
)
_EXPR = st.one_of(_SIMPLE_EXPR, _FUNC_EXPR)
_HOSTILE_STRINGS = st.sampled_from(
    [
        "__import__('os').system('id')",
        "eval('1+1')",
        "exec('print(1)')",
        "globals()['__builtins__']",
        "().__class__.__bases__[0].__subclasses__()",
        "__import__('subprocess').call('ls')",
        "open('/etc/passwd').read()",
        "type.__subclasses__(type)",
        "compile('x','','exec')",
        "vars()['__builtins__'].__import__('os')",
        "getattr(getattr('', '__class__'), '__bases__')[0]",
        "lambda: None",
    ]
)
_IDENT_STRINGS = st.text(
    alphabet=string.ascii_letters + string.digits + "_- ;`$\\(){}[]!@#%^&*",
    min_size=1,
    max_size=100,
)

__all__ = ['math', 'string', 'np', 'pytest', 'given', 'settings', 'HealthCheck', 'st', 'sanitize_ident', 'EquationNeuron', 'UniversalNeuron', '_OPERATORS', '_NUMBERS', '_SAFE_VARS', '_SAFE_FUNCS', '_SIMPLE_EXPR', '_FUNC_EXPR', '_EXPR', '_HOSTILE_STRINGS', '_IDENT_STRINGS']
