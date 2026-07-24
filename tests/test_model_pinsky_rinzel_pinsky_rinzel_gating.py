# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPinskyRinzelGating from former test_model_pinsky_rinzel.py

"""Focused suite: TestPinskyRinzelGating from former test_model_pinsky_rinzel.py."""

from __future__ import annotations

from tests.model_pinsky_rinzel_support import *  # noqa: F403


class TestPinskyRinzelGating:
    def test_gating_variables_bounded(self):
        n = PinskyRinzelNeuron()
        for _ in range(50000):
            n.step(50.0)
        for name, value in (("h", n.h), ("n", n.n), ("s", n.s), ("c", n.c), ("q", n.q)):
            assert 0.0 <= value <= 1.0, f"{name} = {value}"

    def test_sodium_inactivates_at_high_drive(self):
        n = PinskyRinzelNeuron()
        h_initial = n.h
        for _ in range(50000):
            n.step(100.0)
        assert n.h < h_initial
