# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRustNeurons from former test_rust_integration.py

"""Focused suite: TestRustNeurons from former test_rust_integration.py."""

from __future__ import annotations

from tests.rust_integration_support import *  # noqa: F403

class TestRustNeurons:
    @pytest.mark.parametrize(
        ("model", "current"),
        [
            ("Izhikevich", 15.0),
            ("HodgkinHuxleyNeuron", 15.0),
            ("AdExNeuron", 200.0),
            ("LapicqueNeuron", 15.0),
        ],
    )
    def test_neuron_produces_spikes(self, model: str, current: float) -> None:
        cls = getattr(engine, model)
        neuron = cls()
        spikes = sum(neuron.step(current) for _ in range(500))
        assert spikes > 0

    def test_izhikevich_deterministic(self) -> None:
        a = engine.Izhikevich()
        b = engine.Izhikevich()
        sa = [a.step(10.0) for _ in range(100)]
        sb = [b.step(10.0) for _ in range(100)]
        assert sa == sb

    def test_izhikevich_reset(self) -> None:
        n = engine.Izhikevich()
        for _ in range(100):
            n.step(10.0)
        n.reset()
        fresh = engine.Izhikevich()
        assert n.step(0.0) == fresh.step(0.0)

    def test_arcane_neuron_exists(self) -> None:
        n = engine.ArcaneNeuron()
        spike = n.step(5.0)
        assert spike in (0, 1)
