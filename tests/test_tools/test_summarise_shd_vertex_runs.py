# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools/summarise_shd_vertex_runs.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("summarise_shd_vertex_runs", TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_run(
    root: Path,
    name: str,
    *,
    seed: int,
    deployable_test: float,
    rounding_drop: float,
    round_each_epoch: bool = False,
) -> None:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "dcls_version": "max",
                "round_each_epoch": round_each_epoch,
                "sigma_init": 15.0,
                "sigma_final": 0.0,
                "best_fpga_deployable_epoch": 12,
                "fpga_deployable_test_acc": deployable_test,
                "rounding_drop": rounding_drop,
                "last_epoch": 14,
                "last_test_at_sig_final_after_round": deployable_test - 1.0,
            }
        )
        + "\n"
    )
    with (run_dir / "training_log.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "val_acc",
                "fpga_val_acc",
                "test_acc",
            ],
        )
        writer.writeheader()
        writer.writerow({"epoch": 10, "val_acc": 81.0, "fpga_val_acc": 70.0, "test_acc": -1.0})
        writer.writerow({"epoch": 12, "val_acc": 79.0, "fpga_val_acc": 78.0, "test_acc": 68.0})


def test_summarise_runs_aggregates_completed_artifacts(tmp_path: Path) -> None:
    tool = _load_tool()
    _write_run(
        tmp_path,
        "dcls_max_fpga_select_sigma0_seed0",
        seed=0,
        deployable_test=68.9,
        rounding_drop=0.0,
    )
    _write_run(
        tmp_path,
        "dcls_max_fpga_select_sigma0_seed1",
        seed=1,
        deployable_test=72.1,
        rounding_drop=0.2,
        round_each_epoch=True,
    )

    summary = tool.summarise_runs(tmp_path)

    assert summary["schema_version"] == tool.SCHEMA_VERSION
    assert summary["run_count"] == 2
    assert summary["aggregate"]["deployable_test_min"] == 68.9
    assert summary["aggregate"]["deployable_test_max"] == 72.1
    assert summary["aggregate"]["zero_rounding_drop_runs"] == 1
    assert summary["aggregate"]["round_each_epoch_runs"] == 1
    first = summary["runs"][0]
    assert first["best_native_val_epoch"] == 10
    assert first["best_fpga_val_epoch"] == 12
    assert first["test_rows"] == 1


def test_write_outputs_emits_json_csv_and_markdown(tmp_path: Path) -> None:
    tool = _load_tool()
    _write_run(
        tmp_path / "runs",
        "dcls_max_fpga_select_sigma0_seed0",
        seed=0,
        deployable_test=68.9,
        rounding_drop=0.0,
    )
    summary = tool.summarise_runs(tmp_path / "runs")

    json_path, csv_path, md_path = tool.write_outputs(summary, tmp_path / "summary")

    assert json.loads(json_path.read_text())["run_count"] == 1
    assert csv_path.read_text().splitlines()[0].startswith("run,seed,dcls_version")
    report = md_path.read_text()
    assert "SHD Vertex Corrected-Selection Summary" in report
    assert "dcls_max_fpga_select_sigma0_seed0" in report


def test_missing_training_log_still_summarises_config(tmp_path: Path) -> None:
    tool = _load_tool()
    run_dir = tmp_path / "dcls_max_fpga_select_sigma0_seed0"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "seed": 0,
                "dcls_version": "max",
                "round_each_epoch": False,
                "fpga_deployable_test_acc": 68.9,
            }
        )
    )

    summary = tool.summarise_runs(tmp_path)

    run = summary["runs"][0]
    assert run["best_native_val_epoch"] is None
    assert run["best_fpga_val_epoch"] is None
    assert run["best_fpga_deployable_test_acc"] == 68.9
