# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Repetitions of the scaling benchmark are different random networks

"""Run the real dense lane of the scaling benchmark twice and bind its seeds and spread."""

from __future__ import annotations

import pytest

from benchmarks.scaling_benchmark import run_scaling, to_json


def test_each_repetition_has_its_own_seed_and_the_spread_is_recorded(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Two repetitions use consecutive seeds, differ in spikes, and report their rate spread.

    Parameters
    ----------
    capsys : pytest.CaptureFixture
        Swallows the runner's progress lines.
    """
    points = run_scaling([200], 50.0, 2, ["AI"], ["sc_numpy_dense"])
    capsys.readouterr()
    assert len(points) == 1
    point = points[0]
    assert point.seeds == [42, 43]
    assert len(point.runs) == 2
    assert point.runs[0].total_spikes != point.runs[1].total_spikes
    assert point.rate_std > 0.0
    rows = to_json(points, {"host": "test"}, {"sim_ms": 50.0, "repeats": 2})["data"]
    assert isinstance(rows, list)
    row = rows[0]
    assert row["seeds"] == [42, 43] and row["n_runs"] == 2
    assert row["rate_std_hz"] == round(point.rate_std, 3)
    single = run_scaling([200], 50.0, 1, ["AI"], ["sc_numpy_dense"])[0]
    capsys.readouterr()
    assert single.seeds == [42] and single.rate_std == 0.0
    assert single.runs[0].total_spikes == point.runs[0].total_spikes
