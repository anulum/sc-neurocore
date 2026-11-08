# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — iCE40 UP5K calibration tool contracts

"""Refit the committed iCE40 UP5K record and refuse every planted violation of its bound."""

from __future__ import annotations

import copy
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from tools import ice40_up5k_calibration as calibration


@pytest.fixture(scope="module")
def record() -> dict[str, Any]:
    """Return the committed calibration record."""
    return dict(json.loads(calibration.MEASUREMENTS.read_text(encoding="utf-8")))


def _design(record: dict[str, Any], name: str) -> dict[str, Any]:
    return next(d for d in record["designs"] if d["name"] == name)


def test_the_committed_record_holds_and_the_constants_are_current(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--check`` refits the record, finds no violation and matches the generated Rust."""
    assert calibration.main(["--check"]) == 0
    out = capsys.readouterr().out
    assert "calibration holds and the generated constants are current" in out
    assert "dense held4: smallest margin" in out
    assert "dense path regressors" in out


def test_held_out_designs_never_shape_the_model(record: dict[str, Any]) -> None:
    """Removing every held4 row leaves the fitted model unchanged."""
    without = [d for d in record["designs"] if d["role"] != "held4"]
    assert calibration.fit_model(without) == calibration.fit_model(record["designs"])


def test_a_routed_design_above_its_bound_is_refused(record: dict[str, Any]) -> None:
    """One cell above the logic bound, or a path above its bound, is a failure line."""
    model = calibration.fit_model(record["designs"])
    planted = copy.deepcopy(_design(record, "held4_dense_3x3_w18"))
    cells, path = calibration.bound(model, "dense", planted["parameters"])
    planted["core_logic_cells"] = cells
    planted["critical_path_ns_per_seed"] = [path, path, path]
    assert calibration.coverage(model, [planted])[0] == []
    planted["core_logic_cells"] = cells + 1
    planted["critical_path_ns_per_seed"] = [path, path * 1.001, path]
    failures, _ = calibration.coverage(model, [planted])
    assert len(failures) == 2
    assert failures[0] == f"held4_dense_3x3_w18: {cells + 1} logic cells > bound {cells}"


def test_a_harness_overflow_is_judged_by_its_packed_core(record: dict[str, Any]) -> None:
    """An over-capacity design needs a packed core size, and that size within the bound."""
    model = calibration.fit_model(record["designs"])
    planted = copy.deepcopy(_design(record, "fit_dense_12x4_w12"))
    assert planted["status"] == "over capacity"
    assert planted["packed_core_logic_cells"] == 5208
    cells, _ = calibration.bound(model, "dense", planted["parameters"])
    planted["packed_core_logic_cells"] = cells + 1
    assert calibration.coverage(model, [planted])[0] == [
        f"fit_dense_12x4_w12: packed core {cells + 1} logic cells > bound {cells}"
    ]
    del planted["packed_core_logic_cells"]
    assert calibration.coverage(model, [planted])[0] == [
        "fit_dense_12x4_w12: over capacity without a packed core size"
    ]


def test_weight_probes_refuse_a_zero_set_that_costs_more_and_an_empty_group(
    record: dict[str, Any],
) -> None:
    """Zero weights must be the smallest; a group without non-zero weights is a failure."""
    model = calibration.fit_model(record["designs"])
    rows = record["weight_probes"]["rows"]
    assert calibration.probe_failures(model, rows) == []
    planted = copy.deepcopy(rows)
    zero = next(r for r in planted if r["q8_codes"] == "all 0" and r["n_inputs"] == 2)
    zero["core_logic_cells"] = 10_000
    failures = calibration.probe_failures(model, planted)
    assert any("an all-zero weight set is not the smallest" in f for f in failures)
    lonely = [{**zero, "core_logic_cells": 217}]
    assert calibration.probe_failures(model, lonely) == [
        "probe (2, 2, 12): no all-non-zero weight set to compare with"
    ]


def test_the_cited_selection_figures_come_from_the_tool(record: dict[str, Any]) -> None:
    """The margin basis and the path selection table are computed, not copied."""
    worst, name = calibration.dense_logic_cell_leave_one_out(record["designs"])
    assert 1.0 < worst < calibration.DENSE_LOGIC_CELL_MARGIN
    assert name.startswith(("fit_", "held_", "held2_", "held3_"))
    spread = calibration.fit_model(record["designs"])["seed_spread"]
    table = calibration.dense_path_selection(record["designs"], spread)
    assert set(table) == set(calibration.PATH_CANDIDATES)
    assert all(row["loo_fails"] == 0 and row["held4_worst"] < 1.0 for row in table.values())
    chosen = min(table, key=lambda c: table[c]["mean_bound_ratio"])
    assert chosen == "[1, n_in, n_in^2]"
    assert calibration.path_features("dense", {"n_inputs": 3}) == [1.0, 3, 9]


def test_probe_rules_refuse_a_costly_partial_set_and_a_set_above_the_bound(
    record: dict[str, Any],
) -> None:
    """A partly zero set above an all-non-zero one, or a non-zero set above the bound, fails."""
    model = calibration.fit_model(record["designs"])
    rows = [
        r for r in record["weight_probes"]["rows"] if (r["n_inputs"], r["data_width"]) == (2, 12)
    ]
    partial = copy.deepcopy(rows)
    next(r for r in partial if r["q8_codes"] == "one non-zero")["core_logic_cells"] = 10_000
    assert calibration.probe_failures(model, partial) == [
        "probe (2, 2, 12): a partly zero weight set exceeds an all-non-zero one"
    ]
    above = copy.deepcopy(rows)
    cells, _ = calibration.bound(model, "dense", {"n_inputs": 2, "n_neurons": 2, "data_width": 12})
    for r in above:
        if r["q8_codes"] not in ("all 0", "one non-zero"):
            r["core_logic_cells"] = cells + 1
    failures = calibration.probe_failures(model, above)
    assert failures and all(f.endswith(f"{cells + 1} > bound {cells}") for f in failures)
    outside = [{**r, "n_inputs": 1, "n_neurons": 1} for r in above]
    assert calibration.probe_failures(model, outside) == []


def test_write_regenerates_a_stale_file_and_a_broken_record_fails(
    record: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--check`` names a stale file, ``--write`` fixes it, and a violation refuses both."""
    generated = tmp_path / "ice40_calibration.rs"
    generated.write_text("stale\n", encoding="utf-8")
    monkeypatch.setattr(calibration, "GENERATED", generated)
    monkeypatch.setattr(calibration, "REPO_ROOT", tmp_path)
    assert calibration.main(["--check"]) == 1
    assert "ice40_calibration.rs is stale; run with --write" in capsys.readouterr().out
    assert calibration.main(["--write"]) == 0
    assert calibration.main(["--check"]) == 0
    broken = copy.deepcopy(record)
    design = next(d for d in broken["designs"] if d["name"] == "held4_dense_3x3_w18")
    design["core_logic_cells"] = 1_000_000
    measurements = tmp_path / "designs.json"
    measurements.write_text(json.dumps(broken), encoding="utf-8")
    monkeypatch.setattr(calibration, "MEASUREMENTS", measurements)
    capsys.readouterr()
    assert calibration.main(["--write"]) == 1
    out = capsys.readouterr().out
    assert "calibration does not hold:" in out and "held4_dense_3x3_w18: 1000000" in out


def test_check_accepts_another_cpu_rounding_and_refuses_a_real_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Last-digit rounding of another CPU is current; a moved value or changed text is stale."""
    committed = calibration.GENERATED.read_text(encoding="utf-8")
    # Values that an AVX-only Xeon rendered from the same record and libraries.
    rounded = committed.replace("    233.3856488191447,", "    233.38564881914473,").replace(
        "[53.96229729729731, 1.054783783783783];", "[53.9622972972973, 1.0547837837837837];"
    )
    assert rounded != committed
    assert calibration.constants_match(committed, rounded)
    moved = committed.replace("    233.3856488191447,", "    233.3857488191447,")
    assert moved != committed
    assert not calibration.constants_match(committed, moved)
    renamed = committed.replace("pub const LIF_PATH:", "pub const LIF_PATHS:")
    assert not calibration.constants_match(committed, renamed)
    assert not calibration.constants_match(committed, committed.replace("(2, 12)", "(2, 13)"))
    assert calibration.constants_match(
        "pub const X: [f64; 2] = [1.5, 0.0];", "pub const X: [f64; 2] = [\n    1.5,\n    1e-17,\n];"
    )
    generated = tmp_path / "ice40_calibration.rs"
    generated.write_text(rounded, encoding="utf-8")
    monkeypatch.setattr(calibration, "GENERATED", generated)
    monkeypatch.setattr(calibration, "REPO_ROOT", tmp_path)
    assert calibration.main(["--check"]) == 0
    generated.write_text(moved, encoding="utf-8")
    capsys.readouterr()
    assert calibration.main(["--check"]) == 1
    assert "ice40_calibration.rs is stale; run with --write" in capsys.readouterr().out


def test_the_script_entry_point_exits_with_the_check_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run as a script, the tool exits with the status of ``main``."""
    monkeypatch.setattr(sys, "argv", ["ice40_up5k_calibration.py", "--check"])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(Path(calibration.__file__)), run_name="__main__")
    assert raised.value.code == 0
    assert "calibration holds" in capsys.readouterr().out
