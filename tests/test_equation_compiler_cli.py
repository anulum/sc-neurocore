# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation-compiler CLI contract tests

"""Exercise the equation-to-Verilog command through its public CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from sc_neurocore.cli import main


@pytest.mark.parametrize("pipeline", [["--pipeline", "1"], ["--pipeline-points", "mul1"]])
def test_adaptive_precision_refuses_incompatible_pipeline_options(
    pipeline: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The real CLI refuses a pipelined adaptive wrapper and writes no partial output."""
    output = tmp_path / "adaptive"
    with patch(
        "sys.argv",
        [
            "sc-neurocore",
            "compile",
            "dv/dt = -v + I",
            "--adaptive-precision",
            *pipeline,
            "-o",
            str(output),
        ],
    ):
        assert main() == 1
    assert "does not support --pipeline/--pipeline-points" in capsys.readouterr().out
    assert not list(output.rglob("*.v"))


class TestCompileCLI:
    """Pin file generation, naming, and missing-equation behaviour."""

    def test_compile_command_generates_verilog(self, tmp_path: Path) -> None:
        """The compile command writes a complete default-named module."""
        output = tmp_path / "out"
        with patch(
            "sys.argv",
            [
                "sc-neurocore",
                "compile",
                "dv/dt = -(v - E_L)/tau_m + I/C",
                "--threshold",
                "v > -50",
                "--reset",
                "v = -65",
                "--params",
                "E_L=-65,tau_m=10,C=1",
                "--init",
                "v=-65",
                "-o",
                str(output),
            ],
        ):
            result = main()

        verilog_path = output / "sc_equation_neuron.v"
        assert result == 0
        assert verilog_path.is_file()
        verilog = verilog_path.read_text(encoding="utf-8")
        assert "module sc_equation_neuron" in verilog
        assert "endmodule" in verilog

    def test_compile_with_testbench(self, tmp_path: Path) -> None:
        """The testbench option writes the DUT and matching testbench."""
        output = tmp_path / "tb_out"
        with patch(
            "sys.argv",
            [
                "sc-neurocore",
                "compile",
                "dv/dt = I",
                "--init",
                "v=0",
                "--testbench",
                "-o",
                str(output),
                "--module-name",
                "simple",
            ],
        ):
            result = main()

        assert result == 0
        assert (output / "simple.v").is_file()
        assert (output / "tb_simple.v").is_file()

    def test_compile_no_ode_shows_usage(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A missing ODE returns an error and explains the required argument."""
        with patch("sys.argv", ["sc-neurocore", "compile"]):
            result = main()

        assert result == 1
        assert "compile requires an ODE string" in capsys.readouterr().out

    def test_compile_with_custom_module_name(self, tmp_path: Path) -> None:
        """The requested module name controls both filename and declaration."""
        output = tmp_path / "custom"
        with patch(
            "sys.argv",
            [
                "sc-neurocore",
                "compile",
                "dv/dt = -v + I",
                "--module-name",
                "my_custom_lif",
                "-o",
                str(output),
            ],
        ):
            result = main()

        verilog_path = output / "my_custom_lif.v"
        assert result == 0
        assert verilog_path.is_file()
        assert "module my_custom_lif" in verilog_path.read_text(encoding="utf-8")

    def test_compile_passes_format_lowering_draws_and_graded_spike(self, tmp_path: Path) -> None:
        """Width, fraction, map method, v2 lowering, seed, stream and spike value reach the RTL."""
        output = tmp_path / "full"
        with patch(
            "sys.argv",
            [
                "sc-neurocore",
                "compile",
                "dv/dt = sin(I) + 0.1*randn()",
                "--threshold",
                "v > 0.5",
                "--reset",
                "v = 0",
                "--method",
                "map",
                "--data-width",
                "32",
                "--fraction",
                "16",
                "--function-lowering",
                "v2",
                "--rng-seed",
                "5",
                "--random-stream",
                "3",
                "--spike-value",
                "2.0 * v",
                "--testbench",
                "-o",
                str(output),
            ],
        ):
            result = main()

        assert result == 0
        verilog = (output / "sc_equation_neuron.v").read_text(encoding="utf-8")
        assert "parameter [63:0] P_RNG_SEED = 64'd5" in verilog
        assert "parameter [31:0] P_RNG_STREAM = 32'd3" in verilog
        assert "output reg signed [31:0] spike_value_out" in verilog
        assert "// v2 function unit: sin" in verilog
        assert "Q16.16" in verilog
        testbench = (output / "tb_sc_equation_neuron.v").read_text(encoding="utf-8")
        assert "[31:0]" in testbench

    def test_compile_passes_the_exact_random_precision(self, tmp_path: Path) -> None:
        """``--random-precision exact`` emits the 60-bit randn unit; standard is the default."""
        outputs = {}
        for precision in ("standard", "exact"):
            output = tmp_path / precision
            with patch(
                "sys.argv",
                [
                    "sc-neurocore",
                    "compile",
                    "dv/dt = randn()",
                    "--method",
                    "map",
                    "--data-width",
                    "32",
                    "--fraction",
                    "29",
                    "--random-precision",
                    precision,
                    "-o",
                    str(output),
                ],
            ):
                assert main() == 0
            outputs[precision] = (output / "sc_equation_neuron.v").read_text(encoding="utf-8")
        assert outputs["exact"] != outputs["standard"]
        assert "exact mode: 60-bit internals" in outputs["exact"]
        assert "exact mode" not in outputs["standard"]
