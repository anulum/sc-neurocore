# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRyGate from former test_sc_quantum_compiler.py

"""Focused suite: TestRyGate from former test_sc_quantum_compiler.py."""

from __future__ import annotations

from tests.sc_quantum_compiler_support import *  # noqa: F403


class TestRyGate:
    """Ry gate construction and probability encoding contracts."""

    def test_ry_zero_is_identity(self) -> None:
        np.testing.assert_allclose(ry_gate(0.0), np.eye(2), atol=1e-12)

    def test_ry_pi_is_Y_rotation(self) -> None:
        """Ry(pi) should flip |0⟩ to |1⟩."""
        result = ry_gate(np.pi) @ np.array([1, 0], dtype=complex)
        np.testing.assert_allclose(np.abs(result) ** 2, [0, 1], atol=1e-12)

    def test_ry_encodes_probability(self) -> None:
        """Ry(angle) applied to |0⟩ should give P(|1⟩) = p."""
        for p in [0.2, 0.5, 0.8]:
            theta = prob_to_ry_angle(p)
            sv = ry_gate(theta) @ np.array([1, 0], dtype=complex)
            np.testing.assert_allclose(np.abs(sv[1]) ** 2, p, atol=1e-12)
