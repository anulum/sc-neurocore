# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRAMAwareQubitSizing from former test_qc_e2e.py

"""Focused suite: TestRAMAwareQubitSizing from former test_qc_e2e.py."""

from __future__ import annotations

from sc_neurocore.quantum_cognition.bridge_adapter import (
    _get_available_ram,
    compute_max_qubits,
)


class TestRAMAwareQubitSizing:
    """Verify compute_max_qubits respects system RAM."""

    def test_max_qubits_within_bounds(self) -> None:
        """The RAM-derived qubit count stays between 4 and 30."""
        max_q = compute_max_qubits()
        assert 4 <= max_q <= 30, f"max_qubits={max_q} out of [4,30]"

    def test_available_ram_positive(self) -> None:
        """The host's available RAM is detected as a positive amount."""
        ram = _get_available_ram()
        assert ram > 0, "Should detect available RAM"

    def test_safety_factor_effect(self) -> None:
        """A more liberal safety factor never allows fewer qubits than a strict one."""
        q_liberal = compute_max_qubits(safety_factor=0.9)
        q_strict = compute_max_qubits(safety_factor=0.1)
        assert q_liberal >= q_strict, f"Liberal ({q_liberal}) should >= strict ({q_strict})"
