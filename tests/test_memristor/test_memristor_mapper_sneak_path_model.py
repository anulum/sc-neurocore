# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSneakPathModel from former test_memristor_mapper.py

"""Focused suite: TestSneakPathModel from former test_memristor_mapper.py."""

from __future__ import annotations

from memristor_mapper_support import *  # noqa: F403


class TestSneakPathModel:
    def test_worst_case_sneak(self) -> None:
        sneak = SneakPathModel.worst_case_sneak(64, 64, 1e-6, 0.2)
        assert sneak > 0
        assert sneak == pytest.approx(126 * 1e-6 * 0.2)

    def test_larger_array_more_sneak(self) -> None:
        s1 = SneakPathModel.worst_case_sneak(32, 32, 1e-6)
        s2 = SneakPathModel.worst_case_sneak(128, 128, 1e-6)
        assert s2 > s1

    def test_signal_to_sneak_ratio(self) -> None:
        ratio = SneakPathModel.signal_to_sneak_ratio(100e-6, 1e-6, 64, 64)
        assert ratio > 0

    def test_1t1r_no_sneak_needed(self) -> None:
        ratio = SneakPathModel.signal_to_sneak_ratio(100e-6, 1e-6, 4, 4)
        assert ratio > 0
