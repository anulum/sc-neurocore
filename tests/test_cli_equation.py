# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — cli equation tests

"""Exercise cli equation behaviour through the public CLI."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from tests.cli_test_support import run_cli


def test_compile_emit_hls_writes_ap_fixed_cpp(tmp_path: Path) -> None:
    """``compile --emit-hls`` lowers the ODE to synthesisable ap_fixed HLS C++."""
    out_dir = tmp_path / "hls"
    rc = run_cli(
        "compile",
        "dv/dt = (-v + I) / 10.0",
        "--module-name",
        "lif_hls",
        "--emit-hls",
        "--hls-tool",
        "vitis",
        "--hls-threshold",
        "2.5",
        "--output",
        str(out_dir),
    )
    assert rc == 0

    hls_path = out_dir / "lif_hls.hls.cpp"
    assert hls_path.is_file()
    text = hls_path.read_text(encoding="utf-8")
    assert "ap_fixed<16,8>" in text
    assert "#pragma HLS PIPELINE II=1" in text
    assert "void lif_hls(" in text
    # The threshold flows through to the emitted spike comparison.
    assert "2.5" in text
    # The Verilog RTL is still emitted alongside the HLS C++.
    assert (out_dir / "lif_hls.v").is_file()


def test_compile_without_emit_hls_skips_cpp(tmp_path: Path) -> None:
    """Without ``--emit-hls`` no HLS C++ is written; the Verilog path is unchanged."""
    out_dir = tmp_path / "novhls"
    rc = run_cli(
        "compile",
        "dv/dt = (-v + I) / 10.0",
        "--module-name",
        "plain_neuron",
        "--output",
        str(out_dir),
    )
    assert rc == 0
    assert not (out_dir / "plain_neuron.hls.cpp").exists()
    assert (out_dir / "plain_neuron.v").is_file()


@pytest.mark.parametrize(
    "pipeline_arguments",
    [
        ("--pipeline", "auto"),
        ("--pipeline", "1", "--pipeline-points", "_mul0"),
        ("--pipeline-points", "_mul0,_mul1"),
    ],
)
def test_compile_supports_each_pipeline_configuration(
    pipeline_arguments: tuple[str, ...],
    tmp_path: Path,
) -> None:
    """Automatic, staged, and named-point pipeline modes emit valid RTL."""
    output = tmp_path / ("pipeline_" + str(len(list(tmp_path.iterdir()))))

    assert (
        run_cli(
            "compile",
            "dv/dt = v * v + I",
            "--module-name",
            "pipeline_fixture",
            *pipeline_arguments,
            "--output",
            str(output),
        )
        == 0
    )
    assert (output / "pipeline_fixture.v").is_file()


def test_compile_emits_adaptive_precision_wrapper(tmp_path: Path) -> None:
    """Adaptive precision remains wired through the installed command parser."""
    output = tmp_path / "adaptive"

    assert (
        run_cli(
            "compile",
            "dv/dt = (-v + I) / 10.0",
            "--module-name",
            "adaptive_fixture",
            "--adaptive-precision",
            "--lp-width",
            "16",
            "--lp-frac",
            "8",
            "--hp-width",
            "32",
            "--hp-frac",
            "16",
            "--output",
            str(output),
        )
        == 0
    )
    verilog = (output / "adaptive_fixture.v").read_text(encoding="utf-8")
    assert "module adaptive_fixture" in verilog


needs_yosys = pytest.mark.skipif(shutil.which("yosys") is None, reason="Yosys is required")


@needs_yosys
def test_compile_synthesises_the_verilog_it_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--synthesize`` runs Yosys on the module ``compile`` itself wrote.

    Before, it read the ``deploy`` layout (``hdl/*.v`` and a top ``.sv``), which
    ``compile`` never writes, and failed with ``FileNotFoundError``.
    """
    output = tmp_path / "ice40"
    rc = run_cli(
        "compile",
        "dv/dt = -v / 10.0 + I",
        "--threshold",
        "v > 1.0",
        "--reset",
        "v = 0.0",
        "--module-name",
        "synth_fixture",
        "--target",
        "ice40",
        "--synthesize",
        "--output",
        str(output),
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "Synthesis complete" in out
    netlist = json.loads((output / "synth_fixture.json").read_text(encoding="utf-8"))
    assert "synth_fixture" in netlist["modules"]


@needs_yosys
def test_compile_reports_a_failed_synthesis_with_status_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A Yosys run that fails is reported as a failure, not as a missing tool.

    A directory where Yosys must write the netlist makes its ``write_json`` fail.
    """
    output = tmp_path / "blocked"
    (output / "synth_fixture.json").mkdir(parents=True)
    rc = run_cli(
        "compile",
        "dv/dt = I",
        "--module-name",
        "synth_fixture",
        "--target",
        "ice40",
        "--synthesize",
        "--output",
        str(output),
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "Synthesis failed (Yosys errors above)" in out
    assert "Yosys not found" not in out


@pytest.mark.parametrize(
    ("target", "message"),
    [("ice40", "Synthesis skipped (Yosys not found)"), ("artix7", "requires Vivado")],
)
def test_compile_reports_synthesis_it_cannot_run(
    target: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without Yosys, or for a Vivado target, synthesis is skipped with the reason."""
    monkeypatch.setenv("PATH", str(tmp_path / "no-tools"))
    rc = run_cli(
        "compile",
        "dv/dt = I",
        "--target",
        target,
        "--synthesize",
        "--output",
        str(tmp_path / target),
    )
    assert rc == 0
    assert message in capsys.readouterr().out
