# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDtUnderflowGuard from former test_equation_compiler.py

"""Focused suite: TestDtUnderflowGuard from former test_equation_compiler.py."""

from __future__ import annotations

from tests.equation_compiler_support import *  # noqa: F403

class TestDtUnderflowGuard:
    """Q8.8 fixed-point dt underflow detection (issue: silent dead Verilog)."""

    def test_dt_underflow_raises_value_error(self):
        """dt below the smallest representable Q8.8 value must raise."""
        import pytest
        from sc_neurocore.neurons.equation_builder import from_equations
        from sc_neurocore.compiler.equation_compiler import compile_to_verilog

        neuron = from_equations(
            "dv/dt = -v/tau",
            threshold="v > -50",
            reset="v = -65",
            params={"tau": 10.0},
            init={"v": -65.0},
            dt=0.001,  # 0.001 * 256 = 0.256 → 0 in Q8.8
        )
        with pytest.raises(ValueError, match="underflows in Q8.8"):
            compile_to_verilog(neuron)

    def test_dt_underflow_message_actionable(self):
        """The error must name the smallest representable value and suggest a fix."""
        import pytest
        from sc_neurocore.neurons.equation_builder import from_equations
        from sc_neurocore.compiler.equation_compiler import compile_to_verilog

        neuron = from_equations(
            "dv/dt = -v/tau",
            threshold="v > -50",
            reset="v = -65",
            params={"tau": 10.0},
            init={"v": -65.0},
            dt=0.0001,
        )
        with pytest.raises(ValueError) as excinfo:
            compile_to_verilog(neuron)
        msg = str(excinfo.value)
        assert "0.00390625" in msg  # 1/256, the Q8.8 minimum
        assert "dt=1.0" in msg  # one of the suggested values
        assert "fraction=12" in msg  # the alternative format suggestion

    def test_dt_at_minimum_q88_compiles(self):
        """dt exactly equal to 1/256 must compile (smallest valid Q8.8 value)."""
        from sc_neurocore.neurons.equation_builder import from_equations
        from sc_neurocore.compiler.equation_compiler import compile_to_verilog

        neuron = from_equations(
            "dv/dt = -v/tau",
            threshold="v > -50",
            reset="v = -65",
            params={"tau": 10.0},
            init={"v": -65.0},
            dt=1.0 / 256,  # smallest non-zero Q8.8 value
        )
        verilog = compile_to_verilog(neuron)
        # Verify the dt multiplier is non-zero (16'sd1 in Q8.8)
        assert "* 16'sd1;" in verilog or "* 16'sd1\n" in verilog or "16'sd1)" in verilog

    def test_dt_zero_does_not_raise(self):
        """dt=0 is a degenerate but legal case (no advancement) — must not raise."""
        from sc_neurocore.neurons.equation_builder import from_equations
        from sc_neurocore.compiler.equation_compiler import compile_to_verilog

        neuron = from_equations(
            "dv/dt = -v/tau",
            threshold="v > -50",
            reset="v = -65",
            params={"tau": 10.0},
            init={"v": -65.0},
            dt=0.0,
        )
        # No raise; the resulting Verilog will have * 16'sd0 by design
        verilog = compile_to_verilog(neuron)
        assert "16'sd0;" in verilog or "16'sd0\n" in verilog or "16'sd0)" in verilog

    def test_wider_fraction_accepts_smaller_dt(self):
        """Q4.12 (fraction=12) should accept dt values that fail in Q8.8."""
        from sc_neurocore.neurons.equation_builder import from_equations
        from sc_neurocore.compiler.equation_compiler import compile_to_verilog

        neuron = from_equations(
            "dv/dt = -v/tau",
            threshold="v > -50",
            reset="v = -65",
            params={"tau": 10.0},
            init={"v": -65.0},
            dt=0.001,  # would fail in Q8.8 but ok in Q4.12 (0.001*4096 ≈ 4)
        )
        verilog = compile_to_verilog(neuron, fraction=12)
        # 0.001 * 4096 = 4.096 → 4 in Q4.12; assert non-zero dt multiplier
        assert "* 16'sd0;" not in verilog

    def test_cli_default_dt_no_longer_underflows(self):
        """CLI compile with no --dt must succeed (default changed from 0.001 to 1.0)."""
        from unittest.mock import patch
        from sc_neurocore.cli import main

        import tempfile

        with tempfile.TemporaryDirectory() as out:
            with patch(
                "sys.argv",
                [
                    "sc-neurocore",
                    "compile",
                    "dv/dt = -v/tau",
                    "--threshold",
                    "v > -50",
                    "--reset",
                    "v = -65",
                    "--params",
                    "tau=10",
                    "--init",
                    "v=-65",
                    "-o",
                    out,
                    "--module-name",
                    "lif_default_dt",
                ],
            ):
                ret = main()
            assert ret == 0
            import os

            with open(os.path.join(out, "lif_default_dt.v")) as f:
                verilog = f.read()
            # Default dt=1.0 → 16'sd256 in Q8.8
            assert "* 16'sd256" in verilog

    def test_cli_explicit_dt_001_raises_via_value_error(self):
        """Explicit --dt 0.001 must propagate the ValueError through the CLI."""
        import tempfile
        from unittest.mock import patch

        import pytest

        from sc_neurocore.cli import main

        with (
            tempfile.TemporaryDirectory() as out,
            patch(
                "sys.argv",
                [
                    "sc-neurocore",
                    "compile",
                    "dv/dt = -v/tau",
                    "--threshold",
                    "v > -50",
                    "--reset",
                    "v = -65",
                    "--params",
                    "tau=10",
                    "--init",
                    "v=-65",
                    "--dt",
                    "0.001",
                    "-o",
                    out,
                ],
            ),
            pytest.raises(ValueError, match="underflows in Q8.8"),
        ):
            main()
