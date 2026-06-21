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
    verifier = CodeSafetyVerifier()

    assert verifier.verify_code_safety(source) is False


def test_code_safety_verifier_allows_bounded_numpy_source() -> None:
    verifier = CodeSafetyVerifier()

    assert verifier.verify_code_safety("import numpy as np\nx = np.zeros(10)") is True


def test_logic_invariant_returns_false_for_failed_condition() -> None:
    verifier = CodeSafetyVerifier()

    result = verifier.verify_logic_invariant(
        func=lambda x: x * 2,
        input_sample=3,
        expected_condition=lambda output: output == 999,
    )

    assert result is False


def test_logic_invariant_returns_false_for_exception() -> None:
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


def test_logic_invariant_returns_true_when_condition_holds() -> None:
    """The invariant passes when the function output satisfies the condition."""
    verifier = CodeSafetyVerifier()

    result = verifier.verify_logic_invariant(
        func=lambda x: x * 2,
        input_sample=3,
        expected_condition=lambda output: output == 6,
    )

    assert result is True
