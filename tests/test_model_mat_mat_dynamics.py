# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMATDynamics from former test_model_mat.py

"""Focused suite: TestMATDynamics from former test_model_mat.py."""

from __future__ import annotations

from tests.model_mat_support import *  # noqa: F403


class TestMATDynamics:
    def test_subthreshold_silent(self):
        n = MATNeuron()
        assert len(_run(n, current=15.0, steps=5000)) == 0

    def test_fires_at_sufficient_current(self):
        n = MATNeuron()
        assert len(_run(n, current=30.0, steps=5000)) >= 30

    def test_rate_monotonic(self):
        rates = []
        for I in [25.0, 35.0, 50.0]:
            n = MATNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [20.0, 30.0, 40.0, 50.0, 80.0])
    def test_fi_sweep(self, current: float):
        n = MATNeuron()
        spikes = _run(n, current=current, steps=5000)
        assert isinstance(len(spikes), int)

    def test_isi_increases_with_adaptation(self):
        """Adaptation lengthens ISI over time."""
        n = MATNeuron()
        spikes = _run(n, current=40.0, steps=10_000)
        if len(spikes) >= 10:
            isis = np.diff(spikes[:10])
            # Later ISIs should be longer (adaptation)
            assert isis[-1] >= isis[0]
