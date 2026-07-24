# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRacetrackShiftRegister from former test_spintronic_mapper.py

"""Focused suite: TestRacetrackShiftRegister from former test_spintronic_mapper.py."""

from __future__ import annotations

from spintronic_mapper_support import *  # noqa: F403


class TestRacetrackShiftRegister:
    def test_load_and_shift(self):
        rt = RacetrackShiftRegister(n_positions=8)
        rt.load(np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8))
        rt.shift_right()
        assert rt.bits[0] == 0  # shifted in zero
        assert rt.bits[1] == 1  # original bit[0]

    def test_shift_left(self):
        rt = RacetrackShiftRegister(n_positions=4)
        rt.load(np.array([1, 0, 1, 0], dtype=np.int8))
        rt.shift_left()
        assert rt.bits[-1] == 0
        assert rt.bits[0] == 0  # original bit[1]

    def test_shift_energy(self):
        rt = RacetrackShiftRegister(n_positions=8)
        assert rt.shift_energy_fj > 0

    def test_shift_right_injects_error_under_rng(self):
        # With a certain shift-error rate, the rng-driven bit flip path is taken.
        rt = RacetrackShiftRegister(n_positions=8, shift_error_rate=1.0)
        rt.load(np.zeros(8, dtype=np.int8))
        rt.shift_right(rng=np.random.default_rng(0))
        assert int(rt.bits.sum()) == 1  # the single injected flip

    def test_shift_left_injects_error_under_rng(self):
        rt = RacetrackShiftRegister(n_positions=8, shift_error_rate=1.0)
        rt.load(np.zeros(8, dtype=np.int8))
        rt.shift_left(rng=np.random.default_rng(0))
        assert int(rt.bits.sum()) == 1
