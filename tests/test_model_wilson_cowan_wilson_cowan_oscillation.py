# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWilsonCowanOscillation from former test_model_wilson_cowan.py

"""Focused suite: TestWilsonCowanOscillation from former test_model_wilson_cowan.py."""

from __future__ import annotations

from tests.model_wilson_cowan_support import *  # noqa: F403

class TestWilsonCowanOscillation:
    def test_can_oscillate(self):
        """The enrolled feedback regime sustains a non-trivial E limit cycle."""
        n = WilsonCowanUnit(w_ee=16.0, w_ei=12.0, w_ie=15.0, theta=4.0)
        es = []
        for _ in range(5000):
            n.step(1.5)
            es.append(n.e)
        es = np.array(es[1000:])
        mean_e = np.mean(es)
        crossings = np.sum(np.diff(np.sign(es - mean_e)) != 0)
        assert np.ptp(es) > 0.9
        assert crossings >= 30
