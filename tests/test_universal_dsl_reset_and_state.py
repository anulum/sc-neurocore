# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestResetAndState from former test_universal_dsl.py

"""Focused suite: TestResetAndState from former test_universal_dsl.py."""

from __future__ import annotations

from tests.universal_dsl_support import *  # noqa: F403

class TestResetAndState:
    """Test reset and state introspection."""

    def test_reset_restores_initial(self) -> None:
        neuron = UniversalNeuron.from_schema("lif")
        initial = dict(neuron.state)
        for _ in range(100):
            neuron.step(I=30.0)
        neuron.reset()
        assert neuron.state == initial

    def test_state_is_dict(self) -> None:
        neuron = UniversalNeuron.from_schema("fitzhugh_nagumo")
        assert isinstance(neuron.state, dict)
        assert "v" in neuron.state
        assert "w" in neuron.state
