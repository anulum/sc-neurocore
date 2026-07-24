# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestShortTermPlasticity from former test_advanced_plasticity.py

"""Focused suite: TestShortTermPlasticity from former test_advanced_plasticity.py."""

from __future__ import annotations

from tests.advanced_plasticity_support import *  # noqa: F403


class TestShortTermPlasticity:
    def test_returns_scaling(self):
        stp = ShortTermPlasticity(tau_d=200.0, tau_f=600.0, u_se=0.2)
        pre = np.array([1, 0, 1, 0, 0], dtype=np.float64)
        scale = stp.update(pre)
        assert scale.shape == (5,)
        assert np.all(scale >= 0)
        assert np.all(scale <= 1.0)

    def test_depression_on_repeated_spikes(self):
        stp = ShortTermPlasticity(tau_d=50.0, tau_f=600.0, u_se=0.5)
        pre = np.array([1.0])
        s1 = stp.update(pre)[0]
        s2 = stp.update(pre)[0]
        assert s2 < s1
