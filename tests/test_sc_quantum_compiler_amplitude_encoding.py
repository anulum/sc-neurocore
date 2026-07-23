# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAmplitudeEncoding from former test_sc_quantum_compiler.py

"""Focused suite: TestAmplitudeEncoding from former test_sc_quantum_compiler.py."""

from __future__ import annotations

from tests.sc_quantum_compiler_support import *  # noqa: F403

class TestAmplitudeEncoding:
    """Amplitude encoding contracts for SC probabilities."""

    def test_zero_prob(self) -> None:
        sv = sc_prob_to_statevector(0.0)
        np.testing.assert_allclose(sv, [1.0, 0.0])

    def test_one_prob(self) -> None:
        sv = sc_prob_to_statevector(1.0)
        np.testing.assert_allclose(sv, [0.0, 1.0])

    def test_half_prob(self) -> None:
        sv = sc_prob_to_statevector(0.5)
        np.testing.assert_allclose(np.abs(sv) ** 2, [0.5, 0.5])

    def test_born_rule_roundtrip(self) -> None:
        """Encode probability → Born rule should recover it exactly."""
        for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            sv = sc_prob_to_statevector(p)
            p_recovered = statevector_to_prob(sv)
            np.testing.assert_allclose(p_recovered, p, atol=1e-12)

    def test_normalization(self) -> None:
        """State vector should be normalized."""
        for p in [0.1, 0.5, 0.9]:
            sv = sc_prob_to_statevector(p)
            assert np.abs(np.sum(np.abs(sv) ** 2) - 1.0) < 1e-12
