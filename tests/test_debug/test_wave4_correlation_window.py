# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCorrelationWindow from former test_wave4.py

"""Focused suite: TestCorrelationWindow from former test_wave4.py."""

from __future__ import annotations

from wave4_support import *  # noqa: F403


class TestCorrelationWindow:
    def test_mean(self):
        cw = CorrelationWindow(4)
        for v in [0.1, 0.2, 0.3, 0.4]:
            cw.add(v)
        assert cw.mean() == pytest.approx(0.25)

    def test_max(self):
        cw = CorrelationWindow(4)
        for v in [0.1, 0.5, 0.2]:
            cw.add(v)
        assert cw.max() == pytest.approx(0.5)

    def test_count(self):
        cw = CorrelationWindow(10)
        for _ in range(5):
            cw.add(1.0)
        assert cw.count == 5
