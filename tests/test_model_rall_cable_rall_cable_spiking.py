# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRallCableSpiking from former test_model_rall_cable.py

"""Focused suite: TestRallCableSpiking from former test_model_rall_cable.py."""

from __future__ import annotations

from tests.model_rall_cable_support import *  # noqa: F403

class TestRallCableSpiking:
    def test_fewer_compartments_easier_to_spike(self) -> None:
        """Shorter cable (fewer compartments) → less attenuation → more spikes."""
        n2 = RallCableNeuron(n_comp=2, g_ratio=2.0)
        n5 = RallCableNeuron(n_comp=5, g_ratio=2.0)
        s2 = len(_run(n2, current=500.0, steps=50000))
        s5 = len(_run(n5, current=500.0, steps=50000))
        assert s2 > s5, f"n_comp=2: {s2} spikes, n_comp=5: {s5}"

    def test_spikes_with_short_cable(self) -> None:
        """n_comp=2 with strong coupling should produce spikes."""
        n = RallCableNeuron(n_comp=2, g_ratio=2.0)
        spikes = _run(n, current=500.0, steps=50000)
        assert len(spikes) >= 100

    def test_no_spikes_long_cable_weak_coupling(self) -> None:
        """Default (n=5, g_ratio=0.5) with moderate current → no somatic spikes."""
        n = RallCableNeuron()
        spikes = _run(n, current=500.0, steps=50000)
        assert len(spikes) == 0

    def test_somatic_reset_on_spike(self) -> None:
        """After spike, soma resets to v_reset."""
        n = RallCableNeuron(n_comp=2, g_ratio=5.0)
        for _ in range(50000):
            s = n.step(500.0)
            if s == 1:
                assert abs(n.v[0] - n.v_reset) < 1e-6
                break
        else:
            pytest.skip("No spike observed")
