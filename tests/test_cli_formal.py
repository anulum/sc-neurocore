# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — cli formal tests

"""Exercise cli formal behaviour through the public CLI."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from sc_neurocore.formal import validate_formal_network_report
from tests.cli_test_support import run_cli


def test_formal_verify_network_writes_sva_and_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "formal"

    rc = run_cli(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "2",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "16",
        "--max-spikes",
        "3",
        "--output",
        str(out_dir),
    )

    assert rc == 0
    sva_path = out_dir / "dense_lif_frontier_fixture_rate_bound.sv"
    report_path = out_dir / "formal_rate_bound_report.json"
    assert "Formal network verification artifacts written" in capsys.readouterr().out
    assert "a_output0_rate_bound" in sva_path.read_text(encoding="utf-8")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "sc-neurocore.formal-network-rate-bound.v0.1"
    assert report["network"]["name"] == "dense_lif_frontier_fixture"
    assert report["rate_bound"]["window_cycles"] == 16
    assert report["replay"] is None
    assert report["artifacts"]["sva"] == str(sva_path)
    rtl_path = out_dir / "dense_lif_frontier_fixture.v"
    assert report["artifacts"]["rtl"] == str(rtl_path)
    validate_formal_network_report(report, artifact_root=out_dir)
    assert "module dense_lif_frontier_fixture (" in rtl_path.read_text(encoding="utf-8")
    assert "dense_lif_frontier_fixture.v" in (out_dir / "dense_lif_frontier_fixture.sby").read_text(
        encoding="utf-8"
    )


def test_formal_verify_network_replays_safe_trace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trace_path = tmp_path / "safe_trace.json"
    trace_path.write_text("[1, 0, 1, 0, 1, 1]", encoding="utf-8")
    out_dir = tmp_path / "formal"

    rc = run_cli(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "1",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "2",
        "--spike-trace",
        str(trace_path),
        "--output",
        str(out_dir),
    )

    assert rc == 0
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["replay"]["violated"] is False
    assert report["replay"]["cycles_checked"] == 6
    assert "Replay passed" in capsys.readouterr().out


def test_formal_verify_network_replays_unsafe_trace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trace_path = tmp_path / "unsafe_trace.json"
    trace_path.write_text("[1, 0, 1, 1]", encoding="utf-8")
    out_dir = tmp_path / "formal"

    rc = run_cli(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "1",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "2",
        "--spike-trace",
        str(trace_path),
        "--output",
        str(out_dir),
    )

    assert rc == 1
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["replay"]["violated"] is True
    assert report["replay"]["first_violation_cycle"] == 3
    assert report["replay"]["observed_spikes"] == 3
    assert "Replay violation" in capsys.readouterr().out


def test_formal_verify_network_replays_refractory_violation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trace_path = tmp_path / "refractory_unsafe_trace.json"
    trace_path.write_text("[1, 0, 1, 0]", encoding="utf-8")
    out_dir = tmp_path / "formal"

    rc = run_cli(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "1",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "4",
        "--refractory-cycles",
        "3",
        "--spike-trace",
        str(trace_path),
        "--output",
        str(out_dir),
    )

    assert rc == 1
    refractory_path = out_dir / "dense_lif_frontier_fixture_refractory.sv"
    bundle_path = out_dir / "dense_lif_frontier_fixture_formal_bundle.sv"
    assert "a_output0_refractory" in refractory_path.read_text(encoding="utf-8")
    assert "dense_lif_frontier_fixture_refractory_sva" in bundle_path.read_text(encoding="utf-8")
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["refractory"]["refractory_cycles"] == 3
    assert report["refractory_replay"]["violated"] is True
    assert report["refractory_replay"]["first_violation_cycle"] == 2
    assert report["rate_replay"]["violated"] is False
    assert report["artifacts"]["refractory_sva"] == str(refractory_path)
    assert report["artifacts"]["formal_bundle"] == str(bundle_path)
    assert "Refractory violation" in capsys.readouterr().out


def test_formal_verify_network_replays_antagonistic_violation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trace_path = tmp_path / "antagonistic_unsafe_trace.json"
    trace_path.write_text("[[1, 0], [0, 1], [1, 1]]", encoding="utf-8")
    out_dir = tmp_path / "formal"

    rc = run_cli(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "2",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "4",
        "--antagonistic-pair",
        "0,1",
        "--spike-trace",
        str(trace_path),
        "--output",
        str(out_dir),
    )

    assert rc == 1
    antagonistic_path = out_dir / "dense_lif_frontier_fixture_antagonistic.sv"
    bundle_path = out_dir / "dense_lif_frontier_fixture_formal_bundle.sv"
    assert "a_output0_output1_exclusion" in antagonistic_path.read_text(encoding="utf-8")
    assert "dense_lif_frontier_fixture_antagonistic_sva" in bundle_path.read_text(encoding="utf-8")
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["antagonistic_exclusion"]["output_a"] == 0
    assert report["antagonistic_exclusion"]["output_b"] == 1
    assert report["antagonistic_replay"]["violated"] is True
    assert report["antagonistic_replay"]["first_violation_cycle"] == 2
    assert report["rate_replay"]["violated"] is False
    assert report["artifacts"]["antagonistic_sva"] == str(antagonistic_path)
    assert "Antagonistic violation" in capsys.readouterr().out


def test_formal_verify_network_replays_temporal_separation_violation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trace_path = tmp_path / "temporal_unsafe_trace.json"
    trace_path.write_text("[[1, 0], [0, 1], [0, 0]]", encoding="utf-8")
    out_dir = tmp_path / "formal"

    rc = run_cli(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "2",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "4",
        "--temporal-separation",
        "0,1,2",
        "--spike-trace",
        str(trace_path),
        "--output",
        str(out_dir),
    )

    assert rc == 1
    temporal_path = out_dir / "dense_lif_frontier_fixture_temporal_separation.sv"
    bundle_path = out_dir / "dense_lif_frontier_fixture_formal_bundle.sv"
    assert "a_output0_output1_temporal_separation" in temporal_path.read_text(encoding="utf-8")
    assert "dense_lif_frontier_fixture_temporal_separation_sva" in bundle_path.read_text(
        encoding="utf-8"
    )
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["temporal_separation"]["output_a"] == 0
    assert report["temporal_separation"]["output_b"] == 1
    assert report["temporal_separation"]["separation_cycles"] == 2
    assert report["temporal_replay"]["violated"] is True
    assert report["temporal_replay"]["first_violation_cycle"] == 1
    assert report["artifacts"]["temporal_sva"] == str(temporal_path)
    assert "Temporal separation violation" in capsys.readouterr().out


def test_formal_verify_network_replays_population_coactivation_violation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trace_path = tmp_path / "population_unsafe_trace.json"
    trace_path.write_text("[[1, 0, 1], [0, 1, 0]]", encoding="utf-8")
    out_dir = tmp_path / "formal"

    rc = run_cli(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "3",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "4",
        "--coactivation-cap",
        "1",
        "--spike-trace",
        str(trace_path),
        "--output",
        str(out_dir),
    )

    assert rc == 1
    population_path = out_dir / "dense_lif_frontier_fixture_population_coactivation.sv"
    bundle_path = out_dir / "dense_lif_frontier_fixture_formal_bundle.sv"
    assert "a_population_coactivation_cap" in population_path.read_text(encoding="utf-8")
    assert "dense_lif_frontier_fixture_population_coactivation_sva" in bundle_path.read_text(
        encoding="utf-8"
    )
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["population_coactivation"]["max_active_outputs"] == 1
    assert report["population_replay"]["violated"] is True
    assert report["population_replay"]["first_violation_cycle"] == 0
    assert report["population_replay"]["observed_active_outputs"] == 2
    assert report["rate_replay"]["violated"] is False
    assert report["artifacts"]["population_sva"] == str(population_path)
    assert "Population coactivation violation" in capsys.readouterr().out


def test_formal_verify_network_replays_population_silence_violation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trace_path = tmp_path / "population_silence_unsafe_trace.json"
    trace_path.write_text("[[1, 1, 0], [0, 0, 0], [0, 1, 0]]", encoding="utf-8")
    out_dir = tmp_path / "formal"

    rc = run_cli(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "3",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "4",
        "--population-silence",
        "2,2",
        "--spike-trace",
        str(trace_path),
        "--output",
        str(out_dir),
    )

    assert rc == 1
    silence_path = out_dir / "dense_lif_frontier_fixture_population_silence.sv"
    bundle_path = out_dir / "dense_lif_frontier_fixture_formal_bundle.sv"
    assert "a_population_silence_after_coactivation" in silence_path.read_text(encoding="utf-8")
    assert "dense_lif_frontier_fixture_population_silence_sva" in bundle_path.read_text(
        encoding="utf-8"
    )
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["population_silence"]["trigger_active_outputs"] == 2
    assert report["population_silence"]["silence_cycles"] == 2
    assert report["population_silence_replay"]["violated"] is True
    assert report["population_silence_replay"]["first_violation_cycle"] == 2
    assert report["population_silence_replay"]["trigger_cycle"] == 0
    assert report["artifacts"]["population_silence_sva"] == str(silence_path)
    assert "Population silence violation" in capsys.readouterr().out


def test_formal_verify_network_replays_population_inactivity_violation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trace_path = tmp_path / "population_inactivity_unsafe_trace.json"
    trace_path.write_text("[[0, 0], [1, 0], [0, 0], [0, 0], [0, 0]]", encoding="utf-8")
    out_dir = tmp_path / "formal"

    rc = run_cli(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "2",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "8",
        "--max-spikes",
        "8",
        "--population-inactivity",
        "2",
        "--spike-trace",
        str(trace_path),
        "--output",
        str(out_dir),
    )

    assert rc == 1
    inactivity_path = out_dir / "dense_lif_frontier_fixture_population_inactivity.sv"
    bundle_path = out_dir / "dense_lif_frontier_fixture_formal_bundle.sv"
    assert "a_population_inactivity_bound" in inactivity_path.read_text(encoding="utf-8")
    assert "dense_lif_frontier_fixture_population_inactivity_sva" in bundle_path.read_text(
        encoding="utf-8"
    )
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["population_inactivity"]["max_silent_cycles"] == 2
    assert report["population_inactivity_replay"]["violated"] is True
    assert report["population_inactivity_replay"]["first_violation_cycle"] == 4
    assert report["population_inactivity_replay"]["observed_silent_cycles"] == 3
    assert report["artifacts"]["population_inactivity_sva"] == str(inactivity_path)
    assert "Population inactivity violation" in capsys.readouterr().out


def test_formal_verify_network_rejects_missing_action(capsys: pytest.CaptureFixture[str]) -> None:
    rc = run_cli("formal")

    assert rc == 1
    assert "formal verify-network" in capsys.readouterr().out


def test_formal_verify_network_records_missing_symbiyosys(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "formal"

    with mock.patch("sc_neurocore.cli.commands.formal.shutil.which", return_value=None):
        rc = run_cli(
            "formal",
            "verify-network",
            "--module-name",
            "dense_lif_frontier_fixture",
            "--input-width",
            "3",
            "--output-width",
            "1",
            "--state-width",
            "16",
            "--output-index",
            "0",
            "--window-cycles",
            "4",
            "--max-spikes",
            "2",
            "--run-symbiyosys",
            "--output",
            str(out_dir),
        )

    assert rc == 0
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["symbiyosys"]["requested"] is True
    assert report["symbiyosys"]["status"] == "tool_unavailable"
    assert report["symbiyosys"]["returncode"] is None
    assert "SymbiYosys unavailable" in capsys.readouterr().out
    assert (out_dir / "dense_lif_frontier_fixture.sby").exists()


def test_formal_verify_network_runs_symbiyosys_when_available(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "formal"
    completed = subprocess.CompletedProcess(
        args=["/usr/bin/sby", "-f", str(out_dir / "dense_lif_frontier_fixture.sby")],
        returncode=0,
        stdout="PASS\n",
        stderr="",
    )

    with (
        mock.patch("sc_neurocore.cli.commands.formal.shutil.which", return_value="/usr/bin/sby"),
        mock.patch(
            "sc_neurocore.cli.commands.formal.subprocess.run", return_value=completed
        ) as m_run,
    ):
        rc = run_cli(
            "formal",
            "verify-network",
            "--module-name",
            "dense_lif_frontier_fixture",
            "--input-width",
            "3",
            "--output-width",
            "1",
            "--state-width",
            "16",
            "--output-index",
            "0",
            "--window-cycles",
            "4",
            "--max-spikes",
            "2",
            "--run-symbiyosys",
            "--output",
            str(out_dir),
        )

    assert rc == 0
    m_run.assert_called_once()
    assert m_run.call_args.args[0] == [
        "/usr/bin/sby",
        "-f",
        str(out_dir / "dense_lif_frontier_fixture.sby"),
    ]
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["symbiyosys"]["status"] == "passed"
    assert report["symbiyosys"]["returncode"] == 0
    assert report["symbiyosys"]["stdout"] == "PASS\n"
    assert "SymbiYosys passed" in capsys.readouterr().out


def test_formal_verify_network_returns_nonzero_on_symbiyosys_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "formal"
    completed = subprocess.CompletedProcess(
        args=["/usr/bin/sby", "-f", str(out_dir / "dense_lif_frontier_fixture.sby")],
        returncode=1,
        stdout="FAIL\n",
        stderr="assert failed\n",
    )

    with (
        mock.patch("sc_neurocore.cli.commands.formal.shutil.which", return_value="/usr/bin/sby"),
        mock.patch("sc_neurocore.cli.commands.formal.subprocess.run", return_value=completed),
    ):
        rc = run_cli(
            "formal",
            "verify-network",
            "--module-name",
            "dense_lif_frontier_fixture",
            "--input-width",
            "3",
            "--output-width",
            "1",
            "--state-width",
            "16",
            "--output-index",
            "0",
            "--window-cycles",
            "4",
            "--max-spikes",
            "2",
            "--run-symbiyosys",
            "--output",
            str(out_dir),
        )

    assert rc == 1
    report = json.loads((out_dir / "formal_rate_bound_report.json").read_text(encoding="utf-8"))
    assert report["symbiyosys"]["status"] == "failed"
    assert report["symbiyosys"]["returncode"] == 1
    assert report["symbiyosys"]["stderr"] == "assert failed\n"
    assert "SymbiYosys failed" in capsys.readouterr().out


def test_formal_verify_network_rejects_invalid_formal_depth(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = run_cli(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "1",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "2",
        "--formal-depth",
        "0",
        "--output",
        str(tmp_path / "formal"),
    )

    assert rc == 1
    assert "formal-depth" in capsys.readouterr().out


def test_formal_verify_network_rejects_negative_refractory_cycles(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = run_cli(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "1",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "2",
        "--refractory-cycles",
        "-1",
        "--output",
        str(tmp_path / "formal"),
    )

    assert rc == 1
    assert "refractory-cycles" in capsys.readouterr().out


def test_formal_verify_network_rejects_non_positive_population_inactivity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = run_cli(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "1",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "2",
        "--population-inactivity",
        "0",
        "--output",
        str(tmp_path / "formal"),
    )

    assert rc == 1
    assert "population-inactivity" in capsys.readouterr().out
    assert not (tmp_path / "formal").exists()


def test_formal_verify_network_rejects_negative_coactivation_cap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = run_cli(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "2",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "2",
        "--coactivation-cap",
        "-1",
        "--output",
        str(tmp_path / "formal"),
    )

    assert rc == 1
    assert "coactivation-cap" in capsys.readouterr().out
    assert not (tmp_path / "formal").exists()


def test_formal_verify_network_rejects_non_positive_temporal_separation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = run_cli(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "2",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "2",
        "--temporal-separation",
        "0,1,0",
        "--output",
        str(tmp_path / "formal"),
    )

    assert rc == 1
    assert "temporal-separation" in capsys.readouterr().out
    assert not (tmp_path / "formal").exists()


def test_formal_verify_network_rejects_non_positive_population_silence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = run_cli(
        "formal",
        "verify-network",
        "--module-name",
        "dense_lif_frontier_fixture",
        "--input-width",
        "3",
        "--output-width",
        "2",
        "--state-width",
        "16",
        "--output-index",
        "0",
        "--window-cycles",
        "4",
        "--max-spikes",
        "2",
        "--population-silence",
        "0,2",
        "--output",
        str(tmp_path / "formal"),
    )

    assert rc == 1
    assert "population-silence" in capsys.readouterr().out
    assert not (tmp_path / "formal").exists()


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--antagonistic-pair", "0", "two comma-separated"),
        ("--antagonistic-pair", "x,1", "integer output indexes"),
        ("--temporal-separation", "0,1", "must be A,B,CYCLES"),
        ("--temporal-separation", "x,1,2", "integer values"),
        ("--population-silence", "1", "TRIGGER_ACTIVE_OUTPUTS,SILENCE_CYCLES"),
        ("--population-silence", "x,2", "integer values"),
    ],
)
def test_formal_verify_network_rejects_malformed_compound_constraint(
    flag: str,
    value: str,
    message: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Compound constraint parsers reject missing and non-integer fields."""
    assert (
        run_cli(
            "formal",
            "verify-network",
            "--output-width",
            "2",
            flag,
            value,
            "--output",
            str(tmp_path / "formal"),
        )
        == 1
    )
    assert message in capsys.readouterr().out


@pytest.mark.parametrize("payload", ["{}", "{"])
def test_formal_verify_network_rejects_invalid_trace_document(
    payload: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Replay accepts only a syntactically valid JSON list."""
    trace = tmp_path / "trace.json"
    trace.write_text(payload, encoding="utf-8")

    assert (
        run_cli(
            "formal",
            "verify-network",
            "--spike-trace",
            str(trace),
            "--output",
            str(tmp_path / "formal"),
        )
        == 1
    )
    assert "Formal replay invalid" in capsys.readouterr().out


def test_formal_verify_network_reports_schema_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A malformed generated report is rejected before it is persisted."""
    import sc_neurocore.formal as formal

    monkeypatch.setattr(
        formal,
        "validate_formal_network_report",
        mock.Mock(side_effect=ValueError("invalid generated report")),
    )

    assert run_cli("formal", "verify-network", "--output", str(tmp_path / "formal")) == 1
    assert "invalid generated report" in capsys.readouterr().out


def test_formal_verify_network_replays_all_constraints_without_violation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One safe trace exercises every optional network constraint together."""
    trace = tmp_path / "safe_all_constraints.json"
    trace.write_text(
        json.dumps([[1, 0], [0, 0], [0, 1], [0, 0], [1, 0], [0, 0]]),
        encoding="utf-8",
    )

    assert (
        run_cli(
            "formal",
            "verify-network",
            "--output-width",
            "2",
            "--window-cycles",
            "6",
            "--max-spikes",
            "3",
            "--refractory-cycles",
            "1",
            "--antagonistic-pair",
            "0,1",
            "--temporal-separation",
            "0,1,1",
            "--coactivation-cap",
            "1",
            "--population-silence",
            "1,1",
            "--population-inactivity",
            "2",
            "--spike-trace",
            str(trace),
            "--output",
            str(tmp_path / "formal"),
        )
        == 0
    )
    output = capsys.readouterr().out
    for message in (
        "Replay passed",
        "Refractory replay passed",
        "Antagonistic replay passed",
        "Temporal separation replay passed",
        "Population coactivation replay passed",
        "Population silence replay passed",
        "Population inactivity replay passed",
    ):
        assert message in output
