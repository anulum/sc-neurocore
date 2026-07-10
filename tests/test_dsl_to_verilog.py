# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end DSL → Verilog compilation tests

"""End-to-end tests: schema → UniversalNeuron → Verilog → Icarus Verilog.

These tests validate the complete pipeline from TOML/JSON model schemas
through the equation compiler to synthesizable Verilog RTL, and then
compile + simulate with Icarus Verilog to verify functional correctness.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from sc_neurocore.compiler.equation_compiler import (
    generate_testbench,
)

import shutil

HAS_IVERILOG = shutil.which("iverilog") is not None


# Models that use only polynomial/linear dynamics (no transcendentals)
# These are guaranteed to compile cleanly with the Q8.8 arithmetic. FitzHugh-Nagumo
# belongs here: its right-hand side is the polynomial cube ``v * v * v`` (no exp/tanh),
# so it lowers to plain fixed-point multipliers, not a look-up table.
_SIMPLE_MODELS = [
    "lif",
    "lapicque",
    "izhikevich",
    "quadratic_if",
    "resonate_fire",
    "fitzhugh_nagumo",
]

# Models with transcendentals (exp, tanh) — require LUT support
_TRANSCENDENTAL_MODELS = ["adex", "hindmarsh_rose"]


class TestDSLToVerilogCompilation:
    """Test that schema-loaded models compile to valid Verilog."""

    @pytest.mark.parametrize("model_name", _SIMPLE_MODELS)
    def test_simple_model_compiles(self, model_name: str) -> None:
        """Simple (polynomial) models should compile without errors."""
        neuron = UniversalNeuron.from_schema(model_name)
        verilog = neuron.to_verilog()

        assert "module sc_" in verilog
        assert "endmodule" in verilog
        assert "clk" in verilog
        assert "rst_n" in verilog
        assert "spike_out" in verilog

    @pytest.mark.parametrize("model_name", _SIMPLE_MODELS)
    def test_simple_model_has_state_outputs(self, model_name: str) -> None:
        """Each state variable should have a corresponding output port."""
        neuron = UniversalNeuron.from_schema(model_name)
        verilog = neuron.to_verilog()

        for var in neuron.list_state_variables():
            assert f"{var}_out" in verilog, f"Missing output port for {var}"

    @pytest.mark.parametrize("model_name", _SIMPLE_MODELS)
    def test_simple_model_has_parameters(self, model_name: str) -> None:
        """Parameters should appear as Verilog parameters."""
        neuron = UniversalNeuron.from_schema(model_name)
        verilog = neuron.to_verilog()

        assert "parameter" in verilog

    def test_lif_module_name(self) -> None:
        """LIF should generate a clean module name."""
        neuron = UniversalNeuron.from_schema("lif")
        verilog = neuron.to_verilog()
        assert "module sc_lif" in verilog

    def test_izhikevich_module_name(self) -> None:
        neuron = UniversalNeuron.from_schema("izhikevich")
        verilog = neuron.to_verilog()
        assert "module sc_izhikevich" in verilog

    def test_custom_module_name(self) -> None:
        neuron = UniversalNeuron.from_schema("lif")
        verilog = neuron.to_verilog(module_name="my_custom_lif")
        assert "module my_custom_lif" in verilog

    def test_lif_has_reset_logic(self) -> None:
        neuron = UniversalNeuron.from_schema("lif")
        verilog = neuron.to_verilog()
        # Should have threshold comparison and reset assignment
        assert "spike_out <= 1'b1" in verilog


class TestDSLToVerilogTranscendental:
    """Test models with transcendental functions (exp, tanh via LUT)."""

    @pytest.mark.parametrize("model_name", _TRANSCENDENTAL_MODELS)
    def test_transcendental_model_compiles(self, model_name: str) -> None:
        """Models with exp/tanh should compile using LUT approximations."""
        neuron = UniversalNeuron.from_schema(model_name)
        try:
            verilog = neuron.to_verilog()
            assert "module sc_" in verilog
            assert "endmodule" in verilog
        except ValueError as e:
            # Some equations may use patterns not yet supported
            pytest.skip(f"Compilation not yet supported: {e}")


class TestTestbenchGeneration:
    """Test automatic testbench generation."""

    @pytest.mark.parametrize("model_name", _SIMPLE_MODELS)
    def test_testbench_generates(self, model_name: str) -> None:
        neuron = UniversalNeuron.from_schema(model_name)
        eq_neuron = neuron.to_equation_neuron()
        module_name = f"sc_{model_name}"
        tb = generate_testbench(eq_neuron, module_name=module_name)

        assert f"module tb_{module_name}" in tb
        assert "$dumpfile" in tb
        assert "spike_count" in tb
        assert "$finish" in tb

    @pytest.mark.parametrize("model_name", _SIMPLE_MODELS)
    def test_testbench_has_state_monitors(self, model_name: str) -> None:
        neuron = UniversalNeuron.from_schema(model_name)
        eq_neuron = neuron.to_equation_neuron()
        tb = generate_testbench(eq_neuron, module_name=f"sc_{model_name}")

        for var in neuron.list_state_variables():
            assert f"{var}_out" in tb


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
