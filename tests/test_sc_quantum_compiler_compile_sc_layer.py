# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCompileSCLayer from former test_sc_quantum_compiler.py

"""Focused suite: TestCompileSCLayer from former test_sc_quantum_compiler.py."""

from __future__ import annotations

from tests.sc_quantum_compiler_support import *  # noqa: F403


class TestCompileSCLayer:
    """Dense-layer compiler output contracts."""

    def test_output_format(self) -> None:
        weights = np.array([[0.5, 0.3], [0.8, 0.2]])
        inputs = np.array([0.6, 0.4])
        results = compile_sc_layer(weights, inputs)
        assert len(results) == 2
        for r in results:
            assert "neuron_idx" in r
            assert "ry_angles" in r
            assert "expected_output" in r
            assert "quantum_output" in r

    def test_sc_quantum_equivalence(self) -> None:
        """SC and quantum outputs should match (both compute weighted mean)."""
        weights = np.array([[0.5, 0.3, 0.7]])
        inputs = np.array([0.6, 0.4, 0.8])
        results = compile_sc_layer(weights, inputs)
        r = results[0]
        np.testing.assert_allclose(r["expected_output"], r["quantum_output"], atol=1e-10)

    def test_all_outputs_bounded(self) -> None:
        rng = np.random.RandomState(42)
        weights = rng.uniform(0, 1, (4, 6))
        inputs = rng.uniform(0, 1, 6)
        results = compile_sc_layer(weights, inputs)
        for r in results:
            assert 0.0 <= r["expected_output"] <= 1.0
            assert 0.0 <= r["quantum_output"] <= 1.0
