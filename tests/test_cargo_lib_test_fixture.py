# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cargo library-test fixture contracts

"""Real-process contracts for the shared Cargo library-test fixture."""

from __future__ import annotations

from collections.abc import Callable
import subprocess

import pytest


def test_cargo_lib_fixture_surfaces_real_cargo_failure(
    cargo_lib_test: Callable[[str], subprocess.CompletedProcess[str]],
) -> None:
    """Preserve Cargo command and stream diagnostics on a real CLI failure."""
    invalid_filter = "--sc-neurocore-invalid-test-option"

    with pytest.raises(AssertionError) as captured:
        cargo_lib_test(invalid_filter)

    diagnostic = str(captured.value)
    assert "cargo test --no-default-features --jobs 1" in diagnostic
    assert invalid_filter in diagnostic
    assert "exited with code" in diagnostic
    assert "--- stdout ---" in diagnostic
    assert "--- stderr ---" in diagnostic
    assert "unexpected argument" in diagnostic
