# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — cli equation tests

"""Exercise cli equation behaviour through the public CLI."""

from __future__ import annotations

from pathlib import Path

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


@pytest.mark.parametrize(
    ("target", "synthesis_result", "message", "expected_calls"),
    [
        ("ice40", True, "Synthesis complete", 1),
        ("ice40", False, "Yosys not found", 1),
        ("artix7", False, "requires Vivado", 0),
    ],
)
def test_compile_reports_synthesis_outcome(
    target: str,
    synthesis_result: bool,
    message: str,
    expected_calls: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Synthesis status distinguishes success, missing Yosys, and Vivado targets."""
    import sc_neurocore.cli.commands.compile as compile_command

    calls: list[str] = []

    def run_synthesis(
        _output_dir: str,
        observed_target: str,
        _top_module: str,
        _config: dict[str, str],
    ) -> bool:
        calls.append(observed_target)
        return synthesis_result

    monkeypatch.setattr(compile_command, "run_auto_synthesis", run_synthesis)

    assert (
        run_cli(
            "compile",
            "dv/dt = I",
            "--target",
            target,
            "--synthesize",
            "--output",
            str(tmp_path / target),
        )
        == 0
    )
    assert len(calls) == expected_calls
    assert message in capsys.readouterr().out
