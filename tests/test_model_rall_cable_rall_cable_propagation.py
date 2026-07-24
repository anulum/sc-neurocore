# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRallCablePropagation from former test_model_rall_cable.py

"""Focused suite: TestRallCablePropagation from former test_model_rall_cable.py."""

from __future__ import annotations

from tests.model_rall_cable_support import *  # noqa: F403


class TestRallCablePropagation:
    def test_distal_depolarises_more(self) -> None:
        """Current at distal end → distal compartment most depolarised."""
        n = RallCableNeuron()
        for _ in range(5000):
            n.step(100.0)
        assert n.v[-1] > n.v[0], "Distal should be more depolarised than soma"

    def test_signal_attenuates_with_distance(self) -> None:
        """Voltage decreases from distal to soma (passive attenuation)."""
        n = RallCableNeuron(n_comp=5, g_ratio=0.5)
        for _ in range(10000):
            n.step(200.0)
        # Monotonic attenuation: v[4] > v[3] > ... > v[0]
        for i in range(n.n_comp - 1):
            assert n.v[i + 1] >= n.v[i] - 1.0, f"Compartment {i}: {n.v[i]:.2f} > {n.v[i + 1]:.2f}"

    def test_coupling_strength_affects_propagation(self) -> None:
        """Stronger coupling (g_ratio) → less attenuation → more somatic depolarisation."""
        n_weak = RallCableNeuron(n_comp=3, g_ratio=0.1)
        n_strong = RallCableNeuron(n_comp=3, g_ratio=5.0)
        for _ in range(10000):
            n_weak.step(200.0)
            n_strong.step(200.0)
        assert n_strong.v[0] > n_weak.v[0], (
            f"Strong soma={n_strong.v[0]:.2f}, weak soma={n_weak.v[0]:.2f}"
        )
