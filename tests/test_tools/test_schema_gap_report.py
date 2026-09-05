# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for schema-gap reporting tool

from __future__ import annotations

import importlib.util
import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools/schema_gap_report.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("schema_gap_report", TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_live_schema_gap_counts_match_current_checkout() -> None:
    tool = _load_tool()

    report = tool.build_report(REPO)
    modules = tool.collect_model_names(REPO)
    counts = report["counts"]

    assert report["schema_version"] == tool.SCHEMA_VERSION
    assert counts["model_modules"] == len(modules)
    assert counts["schema_models"] == len(tool.collect_schema_names(REPO)) == 80
    assert counts["schema_only_models"] == 0
    assert report["schema_only_models"] == []
    # Every schema stem binds to a live model module, directly or through the alias table.
    assert all(
        name in modules or tool.module_for_schema_stem(name) in modules
        for name in report["schema_models"]
    )
    assert len(report["records"]) == len(modules)
    assert counts["source_modules_without_schema"] == len(report["ranked_enrolment"])
    assert isinstance(counts["net_missing_schema_models"], int)
    assert counts["net_missing_schema_models"] == (
        counts["source_modules_without_schema"] - counts["schema_only_models"]
    )
    assert "stochastic_lif" in {row["model"] for row in report["records"]}
    assert any(
        row["model"] == "stochastic_lif" and row["classification"] == "package_alias"
        for row in report["records"]
    )


def test_live_report_classifies_known_wc_a5_examples() -> None:
    tool = _load_tool()
    report = tool.build_report(REPO)
    records = {row["model"]: row for row in report["records"]}

    assert records["expif"]["schema_present"] is True
    assert records["expif"]["schema_name"] == "exp_if"
    assert records["expif"]["classification"] == "schema_present"
    assert records["ermentrout_kopell_map_neuron"]["schema_present"] is True
    assert records["ermentrout_kopell_map_neuron"]["schema_name"] == (
        "ermentrout_kopell_map_neuron"
    )
    assert records["chialvo_map"]["schema_present"] is True
    assert records["chialvo_map"]["schema_name"] == "chialvo_map"
    assert records["medvedev_map"]["schema_present"] is True
    assert records["medvedev_map"]["schema_name"] == "medvedev_map"
    assert records["ibarz_tanaka_map"]["schema_present"] is True
    assert records["ibarz_tanaka_map"]["schema_name"] == "ibarz_tanaka_map"
    assert records["butera_respiratory"]["classification"] == "schema_present"
    assert records["butera_respiratory"]["priority"] == "P0-schema-present"
    assert records["sc_six_state_thalamocortical"]["classification"] == "rk4_required"
    assert records["sc_six_state_thalamocortical"]["priority"] == ("P3-rk4-or-higher-order-blocked")
    assert records["pinsky_rinzel"]["classification"] == "multi_compartment"
    assert records["akida_neuron"]["classification"] == "event_discrete"
    assert records["astrocyte"]["classification"] == "euler_candidate"


def test_markdown_report_contains_ranked_enrolment_table() -> None:
    tool = _load_tool()
    report = tool.build_report(REPO)
    markdown = tool.render_markdown(report)
    missing = report["counts"]["net_missing_schema_models"]
    without = report["counts"]["source_modules_without_schema"]

    assert f"Net missing schema-DSL models: **{missing}**" in markdown
    assert f"Source modules without a same-name or alias schema: **{without}**" in markdown
    assert "| `P1-euler-schema-candidate` |" in markdown
    assert "| `P3-rk4-or-higher-order-blocked` |" in markdown
    assert "| `P5-out-of-auto-cosim` |" in markdown
    assert "`sc_six_state_thalamocortical`" in markdown
    assert "`stochastic_lif`" in markdown


def test_cli_writes_json_and_markdown_reports(tmp_path: Path) -> None:
    json_path = tmp_path / "schema_gap.json"
    markdown_path = tmp_path / "schema_gap.md"

    json_result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--repo",
            str(REPO),
            "--format",
            "json",
            "--output",
            str(json_path),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    markdown_result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--repo",
            str(REPO),
            "--format",
            "markdown",
            "--output",
            str(markdown_path),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert json_result.returncode == 0, json_result.stderr
    assert markdown_result.returncode == 0, markdown_result.stderr
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    live = _load_tool().build_report(REPO)["counts"]
    assert payload["counts"]["net_missing_schema_models"] == live["net_missing_schema_models"]
    assert (
        payload["counts"]["source_modules_without_schema"] == live["source_modules_without_schema"]
    )
    assert markdown_path.read_text(encoding="utf-8").startswith(
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->"
    )


def test_portable_fixture_counts_unique_schema_stems(tmp_path: Path) -> None:
    tool = _load_tool()
    _write(tmp_path / "src/sc_neurocore/neurons/models/__init__.py", "")
    _write(
        tmp_path / "src/sc_neurocore/neurons/models/lif.py",
        "class LIF:\n    def step(self, current):\n        self.v += current * self.dt\n",
    )
    _write(
        tmp_path / "src/sc_neurocore/neurons/models/rk.py",
        "class RK:\n    def step(self, current):\n        k1 = k2 = k3 = k4 = current\n",
    )
    _write(tmp_path / "src/sc_neurocore/neurons/model_schemas/lif.toml", "[metadata]\n")
    _write(tmp_path / "src/sc_neurocore/neurons/model_schemas/lif.json", "{}\n")

    report = tool.build_report(tmp_path)
    records = {row["model"]: row for row in report["records"]}

    assert report["counts"]["model_modules"] == 2
    assert report["counts"]["schema_models"] == 1
    assert report["counts"]["net_missing_schema_models"] == 1
    assert report["counts"]["source_modules_without_schema"] == 1
    assert records["lif"]["classification"] == "schema_present"
    assert records["rk"]["classification"] == "rk4_required"


def test_classifier_fallbacks_and_priority_helpers() -> None:
    tool = _load_tool()

    classification, evidence = tool.classify_source("opaque", "class Opaque:\n    pass\n")

    assert classification == "source_review_required"
    assert evidence == ["no deterministic integrator evidence matched static rules"]
    assert tool.priority_for("schema_present") == "P0-schema-present"
    assert tool.priority_for("source_review_required") == "P6-source-review-required"


def test_main_prints_json_to_stdout(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    monkeypatch.setattr(sys, "argv", [str(TOOL), "--repo", str(REPO), "--format", "json"])

    tool.main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    live = tool.build_report(REPO)["counts"]
    assert payload["counts"]["net_missing_schema_models"] == live["net_missing_schema_models"]


def test_script_entrypoint_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "entrypoint.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(TOOL), "--repo", str(REPO), "--format", "json", "--output", str(output)],
    )

    runpy.run_path(str(TOOL), run_name="__main__")

    assert json.loads(output.read_text(encoding="utf-8"))["schema_only_models"] == []
