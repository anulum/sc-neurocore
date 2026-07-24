# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAstrocyteNeuron from former test_astrocyte_adapter.py

"""Focused suite: TestAstrocyteNeuron from former test_astrocyte_adapter.py."""

from __future__ import annotations

from tests.astrocyte_adapter_support import *  # noqa: F403


class TestAstrocyteNeuron:
    """Unit tests for the population-compatible astrocyte adapter."""

    def test_step_returns_int(self) -> None:
        """Step returns a binary release-event integer."""
        neuron = AstrocyteNeuron()
        result = neuron.step(0.0)
        assert result in (0, 1)

    def test_v_tracks_ca(self) -> None:
        """Pseudo-voltage mirrors the wrapped calcium state."""
        neuron = AstrocyteNeuron()
        neuron.step(1.0)
        assert neuron.v == neuron.ca

    def test_reset(self) -> None:
        """Reset restores calcium and pseudo-voltage to resting values."""
        neuron = AstrocyteNeuron()
        for _ in range(50):
            neuron.step(5.0)
        neuron.reset()
        assert neuron.ca == 0.05
        assert neuron.v == 0.05

    def test_ip3_accessible(self) -> None:
        """The adapter exposes the wrapped IP3 concentration."""
        neuron = AstrocyteNeuron()
        assert neuron.ip3 > 0
