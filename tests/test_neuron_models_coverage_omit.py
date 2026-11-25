# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — the neuron-model coverage omit list matches its measurement

"""The coverage omit list for ``neurons/models/`` is the measured set of files below 100 %.

The coverage configuration once omitted every neuron model file with the single glob
``*/neurons/models/*``, so no model file was ever held to the 100 % gate. The glob is now a
list of the files that a full sharded run measured below 100 %; every other model file is
measured. The measurement is stored as a fixture with its commit, host, shard layout, absent
native lanes and deviations from CI. These tests hold the list and the measurement to each
other in both directions, and fail when a model file is added without being measured.
"""

from __future__ import annotations

import json
from pathlib import Path
import tomllib
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "src/sc_neurocore/neurons/models"
FIXTURE = ROOT / "tests/fixtures/neuron_models_coverage_ml350_0babf4e1c.json"
PREFIX = "*/neurons/models/"


def _measurement() -> dict[str, Any]:
    """Return the stored measurement."""
    return cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))


def _rows() -> dict[str, dict[str, Any]]:
    """Return the measured rows keyed by file name."""
    files = cast(dict[str, dict[str, Any]], _measurement()["files"])
    return {Path(path).name: row for path, row in files.items()}


def _omit() -> list[str]:
    """Return the coverage run's omit patterns from ``pyproject.toml``."""
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return cast(list[str], configuration["tool"]["coverage"]["run"]["omit"])


def _listed() -> set[str]:
    """Return the model file names that the omit list names one by one."""
    return {pattern[len(PREFIX) :] for pattern in _omit() if pattern.startswith(PREFIX)}


def test_no_glob_omits_the_whole_models_directory() -> None:
    """Every model entry names one file; no pattern covers the directory or several files."""
    entries = [pattern for pattern in _omit() if "neurons/models" in pattern]
    assert entries
    assert all(entry.startswith(PREFIX) and entry.endswith(".py") for entry in entries)
    assert not any(set("*?[") & set(entry[len(PREFIX) :]) for entry in entries)


def test_the_measurement_names_its_commit_host_layout_and_absent_lanes() -> None:
    """The fixture says what was measured, where, how it was sharded and what was missing."""
    measurement = _measurement()
    assert measurement["commit"] == "0babf4e1c"
    assert "ML350" in measurement["host"]
    assert "tools/run_full_cov.sh --shard=K/8" in measurement["shard_layout"]
    lanes = " ".join(cast(list[str], measurement["absent_lanes"]))
    assert all(lane in lanes for lane in ("Julia", "Go", "Mojo"))
    assert measurement["deviations_from_ci"]


def test_every_model_file_was_measured() -> None:
    """A model file added after the measurement fails here until it is measured."""
    present = {path.name for path in MODELS.glob("*.py")}
    assert present == set(_rows())
    assert _measurement()["files_measured"] == len(present)


def test_every_listed_file_was_below_100_percent() -> None:
    """Each omitted model file was measured below 100 %."""
    rows = _rows()
    assert _listed() <= set(rows)
    assert all(
        rows[name]["percent"] < 100.0 and rows[name]["missing_lines"] > 0 for name in _listed()
    )


def test_every_unlisted_file_was_at_100_percent() -> None:
    """Each model file the list leaves out was measured at 100 %, so the gate may hold it there."""
    rows = _rows()
    unlisted = set(rows) - _listed()
    assert unlisted
    assert all(
        rows[name]["percent"] == 100.0 and rows[name]["missing_lines"] == 0 for name in unlisted
    )
    assert len(_listed()) == _measurement()["files_below_100"]


def test_files_listed_only_for_absent_lanes_are_named_consistently() -> None:
    """A file named as listed only for absent lanes misses nothing outside native-lane code."""
    rows = _rows()
    named = {Path(path).name for path in _measurement()["files_listed_only_for_absent_lanes"]}
    assert named <= _listed()
    for name, row in rows.items():
        only_lanes = (
            row["missing_lines"] > 0 and row["missing_lines"] == row["missing_on_native_lane_paths"]
        )
        assert (name in named) == only_lanes, name
