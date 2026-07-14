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
