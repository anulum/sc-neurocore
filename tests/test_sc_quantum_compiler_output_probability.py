# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOutputProbability from former test_sc_quantum_compiler.py

"""Focused suite: TestOutputProbability from former test_sc_quantum_compiler.py."""

from __future__ import annotations

from tests.sc_quantum_compiler_support import *  # noqa: F403


class TestOutputProbability:
    """Circuit output-probability helper contracts."""

    def test_output_probability_matches_manual(self) -> None:
        """output_probability should match P(output_qubit=|1⟩)."""
        circuit = compile_sc_multiply(0.6, 0.4)
        p = circuit.output_probability()
        # Marginal on q1: P(q1=1) = P(01) + P(11) = 0.4
        np.testing.assert_allclose(p, 0.4, atol=1e-10)
