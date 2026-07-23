# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGEDynamics from former test_model_gutkin_ermentrout.py

"""Focused suite: TestGEDynamics from former test_model_gutkin_ermentrout.py."""

from __future__ import annotations

from tests.model_gutkin_ermentrout_support import *  # noqa: F403

class TestGEDynamics:
    def test_fires_under_drive(self) -> None:
        n = GutkinErmentroutNeuron()
        spikes = _run(n, current=5.0, steps=10_000)
        assert len(spikes) >= 10

    def test_subthreshold_silent(self) -> None:
        n = GutkinErmentroutNeuron()
        assert len(_run(n, current=0.5, steps=5000)) == 0

    def test_rate_monotonic(self) -> None:
        rates = []
        for I in [2.0, 5.0, 10.0]:
            n = GutkinErmentroutNeuron()
            rates.append(len(_run(n, current=I, steps=10_000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 2.0, 5.0, 10.0, 20.0])
    def test_fi_sweep(self, current: float) -> None:
        n = GutkinErmentroutNeuron()
        for _ in range(10_000):
            n.step(current)
        assert np.isfinite(n.v)

    def test_voltage_bounded(self) -> None:
        n = GutkinErmentroutNeuron()
        vs = []
        for _ in range(10_000):
            n.step(5.0)
            vs.append(n.v)
        assert min(vs) > -100 and max(vs) < 80
