# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPinskyRinzelFI from former test_model_pinsky_rinzel.py

"""Focused suite: TestPinskyRinzelFI from former test_model_pinsky_rinzel.py."""

from __future__ import annotations

from tests.model_pinsky_rinzel_support import *  # noqa: F403


class TestPinskyRinzelFI:
    def test_quiescent_near_rest(self):
        assert len(_run(PinskyRinzelNeuron(), current_soma=0.0, steps=50000)) <= 5

    @pytest.mark.parametrize("drive", [2.0, 5.0, 20.0])
    def test_low_drive_fires_repetitively(self, drive: float):
        assert len(_run(PinskyRinzelNeuron(), current_soma=drive, steps=50000)) >= 10

    def test_non_monotonic_depolarisation_block(self):
        low = len(_run(PinskyRinzelNeuron(), current_soma=5.0, steps=50000))
        high = len(_run(PinskyRinzelNeuron(), current_soma=200.0, steps=50000))
        assert low > high
        assert high <= 5
