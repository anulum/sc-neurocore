# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for code safety verifier contracts

"""Contracts for code-safety verifier rejection and invariant handling."""

from __future__ import annotations

import pytest

from sc_neurocore.verification.safety import CodeSafetyVerifier


@pytest.mark.parametrize(
    "source",
    [
        "def (broken syntax",
        "import os\nos.system('ls')",
        "from pathlib import Path\nPath('x').unlink()",
        "import os\nos.remove('x')",
        "import subprocess",
        "import socket",
        "from importlib import import_module",
        "import ctypes",
    ],
)
def test_code_safety_verifier_rejects_unsafe_or_invalid_source(source: str) -> None:
    """Unsafe imports, blocked calls, and invalid syntax fail closed."""
    verifier = CodeSafetyVerifier()

    assert verifier.verify_code_safety(source) is False


def test_code_safety_verifier_allows_bounded_numpy_source() -> None:
    """Bounded NumPy setup code remains allowed by the verifier."""
    verifier = CodeSafetyVerifier()

    assert verifier.verify_code_safety("import numpy as np\nx = np.zeros(10)") is True


def test_logic_invariant_returns_false_for_failed_condition() -> None:
    """A predicate miss returns ``False`` instead of raising."""
    verifier = CodeSafetyVerifier()

    result = verifier.verify_logic_invariant(
        func=lambda x: x * 2,
        input_sample=3,
        expected_condition=lambda output: output == 999,
    )

    assert result is False


def test_logic_invariant_returns_false_for_exception() -> None:
    """Callable exceptions are converted into a failed invariant result."""
    verifier = CodeSafetyVerifier()

    result = verifier.verify_logic_invariant(
        func=lambda x: 1 / 0,
        input_sample=1,
        expected_condition=lambda output: True,
    )

    assert result is False


def test_code_safety_verifier_rejects_blocked_builtin_call() -> None:
    """A blocked builtin call (e.g. eval/exec/compile) is rejected at AST level."""
    verifier = CodeSafetyVerifier()

    assert verifier.verify_code_safety("y = eval('1 + 1')") is False


@pytest.mark.parametrize(
    "source",
    [
        "import os",
        "from pathlib import Path",
        "from os import remove\nremove('x')",
        "Path('x').write_text('payload')",
        "socket.socket()",
        "open('x', 'w')",
        "__builtins__.eval('1 + 1')",
        "__builtins__['eval']('1 + 1')",
        "getattr(__builtins__, 'eval')('1 + 1')",
        "from . import local_helper",
    ],
)
def test_code_safety_verifier_rejects_hidden_io_and_dynamic_escape(source: str) -> None:
    """Reject AST-visible file, process, network, and dynamic-builtin escape routes."""
    verifier = CodeSafetyVerifier()

    assert verifier.verify_code_safety(source) is False


def test_code_safety_verifier_allows_unknown_pure_call_shape() -> None:
    """A local pure helper call is not rejected only because it is a call."""
    verifier = CodeSafetyVerifier()

    assert verifier.verify_code_safety("result = clamp_probability(x)") is True


def test_logic_invariant_returns_true_when_condition_holds() -> None:
    """The invariant passes when the function output satisfies the condition."""
    verifier = CodeSafetyVerifier()

    result = verifier.verify_logic_invariant(
        func=lambda x: x * 2,
        input_sample=3,
        expected_condition=lambda output: output == 6,
    )

    assert result is True
