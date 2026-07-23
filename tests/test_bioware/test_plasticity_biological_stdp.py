# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBiologicalSTDP from former test_plasticity.py

"""Focused suite: TestBiologicalSTDP from former test_plasticity.py."""

from __future__ import annotations

from tests.test_bioware.plasticity_support import *  # noqa: F403

class TestBiologicalSTDP:
    def test_potentiation(self) -> None:
        stdp = BiologicalSTDP()
        dw = stdp.compute_dw(5.0)  # post after pre
        assert dw > 0

    def test_depression(self) -> None:
        stdp = BiologicalSTDP()
        dw = stdp.compute_dw(-5.0)  # pre after post
        assert dw < 0

    def test_zero_dt(self) -> None:
        stdp = BiologicalSTDP()
        assert stdp.compute_dw(0.0) == 0.0

    def test_exponential_decay(self) -> None:
        stdp = BiologicalSTDP(tau_plus_ms=20.0)
        dw_near = stdp.compute_dw(1.0)
        dw_far = stdp.compute_dw(40.0)
        assert abs(dw_near) > abs(dw_far)

    def test_update_weight_bounded(self) -> None:
        stdp = BiologicalSTDP(w_max_q88=512, w_min_q88=0)
        w = stdp.update_weight(500, 1.0)
        assert w <= 512
        w = stdp.update_weight(5, -100.0)
        assert w >= 0
