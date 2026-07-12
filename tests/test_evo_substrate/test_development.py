# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CPPN developmental genome tests

"""CPPN developmental genome tests."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.evo_substrate.development import ActivationFunc, CPPNGenome


class TestCPPN:
    def test_query(self) -> None:
        cppn = CPPNGenome()
        val = cppn.query(0.5, 0.5)
        assert 0.0 <= val <= 1.0  # sigmoid output

    def test_weight_matrix(self) -> None:
        cppn = CPPNGenome()
        w = cppn.generate_weight_matrix(4, 4)
        assert w.shape == (4, 4)

    def test_structure(self) -> None:
        cppn = CPPNGenome()
        assert cppn.num_nodes == 3
        assert cppn.num_edges == 2

    @pytest.mark.parametrize(
        ("activation", "value", "expected"),
        [
            (ActivationFunc.SIN, np.pi / 2.0, 1.0),
            (ActivationFunc.GAUSS, 0.0, 1.0),
            (ActivationFunc.STEP, -0.1, 0.0),
            (ActivationFunc.LINEAR, 1.25, 1.25),
        ],
    )
    def test_activation_functions(
        self,
        activation: ActivationFunc,
        value: float,
        expected: float,
    ) -> None:
        cppn = CPPNGenome()
        cppn.nodes[2].activation = activation
        cppn.edges[1].enabled = False
        cppn.edges[0].weight = 1.0

        assert cppn.query(value, 0.0) == pytest.approx(expected)


# ── HW Fitness Tests (Gap 17) ─────────────────────────────────────────
