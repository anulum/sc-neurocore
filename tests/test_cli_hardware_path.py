# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the hardware-path command

"""The ``hardware-path`` command exits 0 on a completed path and 1 on any refusal."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from sc_neurocore.cli.commands.hardware_path import parse_params
from tests.cli_test_support import run_cli

HAS_SYNTHESIS = all(shutil.which(tool) for tool in ("gcc", "iverilog", "vvp", "yosys"))
needs_synthesis = pytest.mark.skipif(
    not HAS_SYNTHESIS, reason="gcc, iverilog, vvp and yosys are required"
)


@needs_synthesis
def test_a_completed_path_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A real model runs to synthesis; the summary and the manifest path are printed."""
    out = tmp_path / "out"
    status = run_cli(
        "hardware-path",
        "LapicqueNeuron",
        "-o",
        str(out),
        "--q-format",
        "Q16.16",
        "--integrator",
        "map",
        "--params",
        "",
        "--dt",
        "0.01",
        "--current",
        "5",
        "--steps",
        "12",
        "--target",
        "xilinx",
        "--module-name",
        "lapicque_demo",
    )
    output = capsys.readouterr().out
    assert status == 0
    assert "Hardware path of LapicqueNeuron: COMPLETED" in output
    assert f"Manifest: {out / 'hardware_path.json'}" in output
    manifest = json.loads((out / "hardware_path.json").read_text(encoding="utf-8"))
    assert manifest["request"]["current"] == 5.0
    assert manifest["request"]["n_steps"] == 12
    assert "rtl/lapicque_demo.v" in manifest["artifacts"]


def test_a_refusal_exits_one_and_names_the_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown model is refused at the capability gate; the record is named."""
    out = tmp_path / "out"
    assert run_cli("hardware-path", "NoSuchNeuron", "-o", str(out)) == 1
    output = capsys.readouterr().out
    assert "Error: hardware path of NoSuchNeuron refused at capabilities" in output
    assert "unknown-model: 'NoSuchNeuron' is not a catalogue model" in output
    assert f"Record: {out / 'hardware_path.json'}" in output


def test_an_occupied_output_is_refused_without_a_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-empty output directory is refused before anything is written."""
    (tmp_path / "keep").write_text("x", encoding="utf-8")
    assert run_cli("hardware-path", "LapicqueNeuron", "-o", str(tmp_path)) == 1
    output = capsys.readouterr().out
    assert "output-not-empty" in output
    assert "Record:" not in output


@pytest.mark.parametrize(
    "argv",
    [
        ("hardware-path",),
        ("hardware-path", "LapicqueNeuron"),
        ("hardware-path", "LapicqueNeuron", "AdExNeuron", "-o", "unused"),
    ],
)
def test_one_model_and_an_output_are_required(
    argv: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    """Without exactly one model and an output directory the command exits 1."""
    assert run_cli(*argv) == 1
    assert "needs exactly one MODEL and --output DIR" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ("tau", "'tau' is not name=value"),
        ("=1", "'=1' is not name=value"),
        ("tau=abc", "'tau' has a value that is not a number"),
        ("tau=inf", "'tau' is not finite"),
    ],
)
def test_malformed_parameter_overrides_exit_one(
    tmp_path: Path, params: str, message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed ``--params`` is refused before the path starts."""
    status = run_cli(
        "hardware-path", "LapicqueNeuron", "-o", str(tmp_path / "o"), "--params", params
    )
    assert status == 1
    assert message in capsys.readouterr().out
    assert not (tmp_path / "o").exists()


def test_parameter_overrides_parse_into_numbers() -> None:
    """Pairs separated by commas parse to floats; empty input is no override."""
    assert parse_params("tau=10, R = 2.5,") == {"tau": 10.0, "R": 2.5}
    assert parse_params(None) == {}


def test_list_names_supported_and_refused_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--list`` prints each model's verdict on this host and the count."""
    monkeypatch.setenv("PATH", str(tmp_path))
    assert run_cli("hardware-path", "--list", "LapicqueNeuron", "ATypeKNeuron") == 0
    output = capsys.readouterr().out
    assert (
        "LapicqueNeuron: refused, cosimulate: co-simulation needs iverilog, vvp, gcc, "
        "which is not installed"
    ) in output
    assert "ATypeKNeuron: refused, compile: the model has no canonical schema" in output
    assert "0 of 2 models supported on this host" in output


@needs_synthesis
def test_list_on_a_host_with_the_tools(capsys: pytest.CaptureFixture[str]) -> None:
    """With the tools installed a supported model lists its co-simulated combinations."""
    assert run_cli("hardware-path", "--list", "LapicqueNeuron") == 0
    output = capsys.readouterr().out
    assert "LapicqueNeuron: supported (map@Q8.8, map@Q16.16)" in output
    assert "1 of 1 models supported on this host" in output


def test_help_lists_the_options(capsys: pytest.CaptureFixture[str]) -> None:
    """The command's help names its options and the list mode."""
    with pytest.raises(SystemExit, match="0"):
        run_cli("hardware-path", "--help")
    output = capsys.readouterr().out
    for option in ("--list", "--q-format", "--integrator", "--params", "--steps", "--target"):
        assert option in output
