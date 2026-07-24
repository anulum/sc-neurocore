# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStepCurrent from former test_network_monitors_stimulus.py

"""Focused suite: TestStepCurrent from former test_network_monitors_stimulus.py."""

from __future__ import annotations

from tests.network_monitors_stimulus_support import *  # noqa: F403


class TestStepCurrent:
    def test_zero_outside_window(self):
        sc = StepCurrent(onset=100, offset=200, amplitude=5.0)
        assert sc.get_current(50) == 0.0
        assert sc.get_current(250) == 0.0

    def test_amplitude_inside_window(self):
        sc = StepCurrent(onset=100, offset=200, amplitude=5.0)
        assert sc.get_current(150) == 5.0

    def test_onset_inclusive(self):
        sc = StepCurrent(onset=10, offset=20, amplitude=1.0)
        assert sc.get_current(10) == 1.0

    def test_offset_exclusive(self):
        sc = StepCurrent(onset=10, offset=20, amplitude=1.0)
        assert sc.get_current(20) == 0.0

    def test_negative_amplitude(self):
        sc = StepCurrent(onset=0, offset=100, amplitude=-3.0)
        assert sc.get_current(50) == -3.0
