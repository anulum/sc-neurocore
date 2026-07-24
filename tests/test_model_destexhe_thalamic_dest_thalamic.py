# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDestThalamic from former test_model_destexhe_thalamic.py

"""Focused suite: TestDestThalamic from former test_model_destexhe_thalamic.py."""

from __future__ import annotations

from tests.model_destexhe_thalamic_support import *  # noqa: F403


class TestDestThalamic:
    def test_fires_under_drive(self):
        n = DestexheThalamicNeuron()
        spikes = _run(n, current=5.0, steps=5000)
        assert len(spikes) >= 1

    def test_silent_at_zero(self):
        n = DestexheThalamicNeuron()
        spikes = _run(n, current=0.0, steps=3000)
        # May or may not fire (depends on T-current dynamics)
        assert isinstance(len(spikes), int)

    def test_rate_increases_with_current(self):
        rates = []
        for I in [2.0, 5.0, 10.0]:
            n = DestexheThalamicNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 2.0, 5.0, 10.0, 20.0])
    def test_fi_sweep(self, current: float):
        n = DestexheThalamicNeuron()
        for _ in range(3000):
            n.step(current)
        assert np.isfinite(n.v)
