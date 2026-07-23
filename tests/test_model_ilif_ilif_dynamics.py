# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestILIFDynamics from former test_model_ilif.py

"""Focused suite: TestILIFDynamics from former test_model_ilif.py."""

from __future__ import annotations

from tests.model_ilif_support import *  # noqa: F403

class TestILIFDynamics:
    def test_fires_under_drive(self):
        n = InhibitoryLIFNeuron()
        spikes = _run(n, current=5.0, steps=5000)
        assert len(spikes) >= 100

    def test_subthreshold_silent(self):
        n = InhibitoryLIFNeuron()
        assert len(_run(n, current=0.05, steps=5000)) == 0

    def test_rate_monotonic(self):
        rates = []
        for I in [2.0, 5.0, 10.0]:
            n = InhibitoryLIFNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 1.0, 5.0, 10.0])
    def test_fi_sweep(self, current: float):
        n = InhibitoryLIFNeuron()
        for _ in range(5000):
            n.step(current)
        assert np.isfinite(n.v)
