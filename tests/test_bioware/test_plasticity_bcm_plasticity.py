# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBCMPlasticity from former test_plasticity.py

"""Focused suite: TestBCMPlasticity from former test_plasticity.py."""

from __future__ import annotations

from tests.test_bioware.plasticity_support import *  # noqa: F403


class TestBCMPlasticity:
    def test_threshold_update(self) -> None:
        bcm = BCMPlasticity()
        bcm.update_theta(10.0, dt_ms=10.0)
        assert bcm.theta > 0

    def test_ltp_above_threshold(self) -> None:
        bcm = BCMPlasticity()
        bcm.theta = 5.0
        dw = bcm.compute_dw(10.0, 10.0)  # post > theta
        assert dw > 0

    def test_ltd_below_threshold(self) -> None:
        bcm = BCMPlasticity()
        bcm.theta = 20.0
        dw = bcm.compute_dw(10.0, 10.0)  # post < theta
        assert dw < 0

    def test_weight_bounded(self) -> None:
        bcm = BCMPlasticity(w_max_q88=512, w_min_q88=0)
        bcm.theta = 0.0
        w = bcm.update_weight(510, 100.0, 100.0)
        assert w <= 512
