# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNGDynamics from former test_model_neurogrid.py

"""Focused suite: TestNGDynamics from former test_model_neurogrid.py."""

from __future__ import annotations

from tests.model_neurogrid_support import *  # noqa: F403

class TestNGDynamics:
    def test_subthreshold_silent(self) -> None:
        n = NeuroGridNeuron()
        assert len(_run(n, current=20.0, steps=5000)) == 0

    def test_fires_under_drive(self) -> None:
        n = NeuroGridNeuron()
        assert len(_run(n, current=100.0, steps=10_000)) >= 5

    def test_rate_monotonic(self) -> None:
        rates = []
        for I in [50.0, 100.0, 200.0]:
            n = NeuroGridNeuron()
            rates.append(len(_run(n, current=I, steps=10_000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 50.0, 100.0, 150.0, 200.0])
    def test_fi_sweep(self, current: float) -> None:
        n = NeuroGridNeuron()
        for _ in range(10_000):
            n.step(current)
        assert np.isfinite(n.v_s) and np.isfinite(n.v_d)
