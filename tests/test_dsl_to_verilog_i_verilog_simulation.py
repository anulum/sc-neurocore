# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIVerilogSimulation from former test_dsl_to_verilog.py

"""Focused suite: TestIVerilogSimulation from former test_dsl_to_verilog.py."""

from __future__ import annotations

from tests.dsl_to_verilog_support import *  # noqa: F403


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestIVerilogSimulation:
    """End-to-end: schema → Verilog → Icarus compile → simulation.

    Verifies that the equation compiler generates Verilog that
    compiles and simulates correctly with Icarus Verilog.
    """

    @pytest.mark.parametrize("model_name", _SIMPLE_MODELS)
    def test_iverilog_compiles_and_runs(self, model_name: str) -> None:
        """End-to-end: schema → Verilog → Icarus → simulation."""
        neuron = UniversalNeuron.from_schema(model_name)
        eq_neuron = neuron.to_equation_neuron()
        module_name = f"sc_{model_name}"

        verilog = neuron.to_verilog(module_name=module_name)
        tb = generate_testbench(
            eq_neuron,
            module_name=module_name,
            n_steps=50,
            input_current=5.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            rtl_path = Path(tmpdir) / f"{module_name}.v"
            tb_path = Path(tmpdir) / f"tb_{module_name}.v"
            out_path = Path(tmpdir) / f"tb_{module_name}"

            rtl_path.write_text(verilog)
            tb_path.write_text(tb)

            # Compile
            result = subprocess.run(
                ["iverilog", "-g2012", "-o", str(out_path), str(rtl_path), str(tb_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, f"iverilog compile failed:\n{result.stderr}"

            # Simulate
            result = subprocess.run(
                ["vvp", str(out_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, f"vvp simulation failed:\n{result.stderr}"
            assert "Simulation complete" in result.stdout, f"Unexpected output:\n{result.stdout}"
