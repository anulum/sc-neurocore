# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (input_validation) from former test_cli_formal.py

from __future__ import annotations

from tests.cli_formal_support import *  # noqa: F403


def test_formal_verify_network_rejects_missing_action(capsys: pytest.CaptureFixture[str]) -> None:
    rc = run_cli("formal")

    assert rc == 1
    assert "formal verify-network" in capsys.readouterr().out


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
