# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMLDynamics from former test_model_morris_lecar.py

"""Focused suite: TestMLDynamics from former test_model_morris_lecar.py."""

from __future__ import annotations

from tests.model_morris_lecar_support import *  # noqa: F403


class TestMLDynamics:
    @pytest.mark.parametrize("current", [50.0, 80.0, 100.0, 120.0, 150.0])
    def test_fi_sweep(self, current: float):
        n = MorrisLecarNeuron()
        for _ in range(20_000):
            n.step(current)
        assert np.isfinite(n.v)

    def test_regular_isi_in_band(self):
        n = MorrisLecarNeuron()
        spikes = _run(n, current=100.0, steps=50_000)
        if len(spikes) >= 10:
            isis = np.diff(spikes[3:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv < 0.1

    def test_upward_crossing_only(self):
        n = MorrisLecarNeuron()
        prev_v = n.v
        for _ in range(20_000):
            spike = n.step(100.0)
            if spike == 1:
                assert prev_v < n.v_threshold
            prev_v = n.v
