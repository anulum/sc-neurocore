# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExpIFFI from former test_model_expif.py

"""Focused suite: TestExpIFFI from former test_model_expif.py."""

from __future__ import annotations

from tests.model_expif_support import *  # noqa: F403


class TestExpIFFI:
    @pytest.mark.parametrize(
        ("current", "expected"),
        [(0.0, 0), (5.0, 0), (10.0, 1), (20.0, 2), (50.0, 5), (100.0, 9)],
    )
    def test_enrolled_1000_step_event_goldens(self, current: float, expected: int) -> None:
        assert len(_run(ExpIFNeuron(), current=current, steps=1000)) == expected

    def test_subthreshold_current_is_silent(self) -> None:
        assert _run(ExpIFNeuron(), current=1.0, steps=10_000) == []

    def test_suprathreshold_current_fires(self) -> None:
        assert len(_run(ExpIFNeuron(), current=20.0, steps=10_000)) == 23

    def test_monotonic_fi_on_enrolled_operating_points(self) -> None:
        counts = [
            len(_run(ExpIFNeuron(), current=current, steps=1000))
            for current in (0.0, 5.0, 10.0, 20.0, 50.0, 100.0)
        ]
        assert counts == [0, 0, 1, 2, 5, 9]

    def test_constant_drive_has_regular_interspike_intervals(self) -> None:
        spikes = _run(ExpIFNeuron(), current=50.0, steps=10_000)
        intervals = np.diff(spikes[3:]).astype(float)
        assert intervals.size > 10
        assert float(np.std(intervals) / np.mean(intervals)) < 0.05
