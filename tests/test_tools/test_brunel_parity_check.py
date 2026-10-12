# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The parity check accepts and refuses real recordings by the published tolerance

"""Run the parity check on the committed recording and on altered copies of it."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from tools.brunel_parity_check import RATE_TOLERANCE, main, rate_ratios

ROOT = Path(__file__).resolve().parents[2]
RECORDING = ROOT / "benchmarks/results/rust_scaling_benchmark.json"


def _recording() -> dict[str, Any]:
    """Return the committed scaling recording."""
    loaded: dict[str, Any] = json.loads(RECORDING.read_text(encoding="utf-8"))
    return loaded


def _written(path: Path, recording: object) -> str:
    """Write a recording and return its path as a command-line argument."""
    path.write_text(json.dumps(recording), encoding="utf-8")
    return str(path)


def test_the_committed_recording_keeps_the_published_agreement(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Three matched sizes of the committed recording are within the tolerance.

    Parameters
    ----------
    capsys : pytest.CaptureFixture
        Captures the report lines.
    """
    ratios = rate_ratios(_recording())
    assert sorted(ratios) == [1000, 5000, 10000]
    assert all(abs(ratio - 1.0) <= RATE_TOLERANCE for ratio in ratios.values())
    assert main([str(RECORDING), "--minimum-sizes", "3"]) == 0
    assert capsys.readouterr().out.count(" ok") == 3
    assert main([str(RECORDING), "--minimum-sizes", "4"]) == 2


def test_a_rate_outside_the_tolerance_fails_and_one_just_inside_passes(tmp_path: Path) -> None:
    """The boundary is the published tolerance, in both directions.

    Parameters
    ----------
    tmp_path : Path
        Directory for the altered recordings.
    """
    base = _recording()
    reference = {
        row["n_neurons"]: row["rate_mean_hz"]
        for row in base["data"]
        if row["simulator"] == "brian2"
    }
    for factor, expected in ((1.049, 0), (0.951, 0), (1.06, 1), (0.94, 1)):
        altered = copy.deepcopy(base)
        for row in altered["data"]:
            if row["simulator"] == "sc_numpy_dense":
                row["rate_mean_hz"] = reference[row["n_neurons"]] * factor
        assert main([_written(tmp_path / f"r{factor}.json", altered)]) == expected, factor


def test_an_unusable_recording_is_reported_not_passed(tmp_path: Path) -> None:
    """Missing rows, fields, lanes, a zero reference and a missing file all stop the check.

    Parameters
    ----------
    tmp_path : Path
        Directory for the altered recordings.
    """
    base = _recording()
    without_reference = {"data": [row for row in base["data"] if row["simulator"] != "brian2"]}
    without_field = {"data": [{"simulator": "brian2", "n_neurons": 1000}]}
    zero = copy.deepcopy(base)
    for row in zero["data"]:
        if row["simulator"] == "brian2":
            row["rate_mean_hz"] = 0.0
    for name, recording in (
        ("list", []),
        ("rows", {"data": "none"}),
        ("lanes", without_reference),
        ("field", without_field),
        ("zero", zero),
    ):
        with pytest.raises(ValueError):
            rate_ratios(recording)
        assert main([_written(tmp_path / f"{name}.json", recording)]) == 2
    assert main([str(tmp_path / "absent.json")]) == 2


def test_a_second_regime_of_a_size_does_not_replace_the_first() -> None:
    """Each size is judged once, in its first regime, against the reference of that regime."""
    rows = [
        {"simulator": "brian2", "n_neurons": 100, "regime": "AI", "rate_mean_hz": 10.0},
        {"simulator": "sc_numpy_dense", "n_neurons": 100, "regime": "AI", "rate_mean_hz": 10.2},
        {"simulator": "brian2", "n_neurons": 100, "regime": "SR", "rate_mean_hz": 50.0},
        {"simulator": "sc_numpy_dense", "n_neurons": 100, "regime": "SR", "rate_mean_hz": 90.0},
        {"simulator": "sc_numpy_dense", "n_neurons": 200, "regime": "AI", "rate_mean_hz": 9.0},
    ]
    assert rate_ratios({"data": rows}) == {100: 1.02}


def test_the_rerun_recording_is_what_the_readme_says_it_is() -> None:
    """The cited rerun has eight distinct seeds per lane, one second simulated, and the cited ratios."""
    rerun = json.loads(
        (ROOT / "benchmarks/results/brunel_parity_rerun_2026-10-12.json").read_text(
            encoding="utf-8"
        )
    )
    assert rerun["params"]["sim_ms"] == 1000.0 and rerun["params"]["repeats"] == 8
    assert rerun["timestamp"].startswith("2026-10-12")
    for row in rerun["data"]:
        assert row["n_runs"] == 8 and len(set(row["seeds"])) == 8
        assert 0.01 <= row["rate_std_hz"] / row["rate_mean_hz"] <= 0.045
    ratios = rate_ratios(rerun)
    assert sorted(ratios) == [1000, 5000]
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    assert (
        f"rate ratios of {ratios[1000]:.3f} at 1,000 neurons and {ratios[5000]:.3f} at 5,000"
        in readme
    )
    assert "one loaded host, 1 s simulated, eight seeds per lane" in readme
    assert "varied by about 1 to 4 %" in readme
