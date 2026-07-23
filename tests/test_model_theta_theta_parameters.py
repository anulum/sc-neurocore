# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThetaParameters from former test_model_theta.py

"""Focused suite: TestThetaParameters from former test_model_theta.py."""

from __future__ import annotations

from tests.model_theta_support import *  # noqa: F403

class TestThetaParameters:
    @pytest.mark.parametrize("dt", [0.005, 0.01, 0.02])
    def test_dt_stability(self, dt: float) -> None:
        n = ThetaNeuron(dt=dt)
        for _ in range(50000):
            n.step(2.0)
        assert np.isfinite(n.theta)

    def test_dt_affects_isi_steps_not_time(self) -> None:
        """Finer dt → more steps per ISI, but ISI_time stays the same."""
        n1 = ThetaNeuron(dt=0.01)
        n2 = ThetaNeuron(dt=0.005)
        s1 = _run(n1, current=1.0, steps=100000)
        s2 = _run(n2, current=1.0, steps=200000)
        if len(s1) > 5 and len(s2) > 5:
            isi_time_1 = np.mean(np.diff(s1[2:])) * 0.01
            isi_time_2 = np.mean(np.diff(s2[2:])) * 0.005
            assert abs(isi_time_1 - isi_time_2) < 0.1
