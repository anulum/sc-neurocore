# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (constraint_replays) from former test_cli_formal.py

from __future__ import annotations

from tests.cli_formal_support import *  # noqa: F403


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
