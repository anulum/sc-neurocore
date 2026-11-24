# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWendlingSigmoid from former test_model_wendling.py

"""Focused suite: TestWendlingSigmoid from former test_model_wendling.py."""

from __future__ import annotations

import pytest
from sc_neurocore.neurons.models.wendling import (
    WendlingNeuron,
)


class TestWendlingSigmoid:
    """The sigmoid of the WendlingNeuron's population model."""

    def test_sigmoid_formula(self) -> None:
        """S(x) = 2·e0 / (1 + exp(r·(v0 - x)))."""
        n = WendlingNeuron()
        # At x = v0: S = 2·e0 / (1+exp(0)) = 2·2.5/2 = 2.5
        s_at_v0 = float(n._sigmoid(n.v0))
        assert abs(s_at_v0 - n.e0) < 1e-10

    def test_sigmoid_monotonic(self) -> None:
        """The population sigmoid does not fall as its input rises."""
        n = WendlingNeuron()
        vals = [float(n._sigmoid(x)) for x in [-10, 0, 6, 10, 20]]
        assert all(vals[j] <= vals[j + 1] for j in range(len(vals) - 1))

    def test_sigmoid_bounded(self) -> None:
        """The population sigmoid stays inside [0, 2·e0]: S(x) ∈ [0, 2·e0]."""
        n = WendlingNeuron()
        for x in [-100, 0, 6, 100]:
            s = float(n._sigmoid(x))
            assert 0.0 <= s <= 2 * n.e0 + 0.01

    def test_sigmoid_extreme_inputs_remain_bounded(self) -> None:
        """Sigmoid extreme inputs remain bounded."""
        n = WendlingNeuron()

        assert 0.0 <= n._sigmoid(-1e6) < 1e-100
        assert n._sigmoid(1e6) == pytest.approx(2 * n.e0)
