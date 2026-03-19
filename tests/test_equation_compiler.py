# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for equation → Verilog compiler

"""Tests for equation_compiler: ODE strings → synthesizable Verilog RTL."""

from sc_neurocore.neurons.equation_builder import EquationNeuron, from_equations
from sc_neurocore.compiler.equation_compiler import (
    Q88,
    compile_to_verilog,
    equation_to_fpga,
)


class TestQ88:
    def test_encode_zero(self):
        q = Q88()
        assert q.encode(0.0) == 0

    def test_encode_one(self):
        q = Q88()
        assert q.encode(1.0) == 256

    def test_encode_half(self):
        q = Q88()
        assert q.encode(0.5) == 128

    def test_encode_negative(self):
        q = Q88()
        raw = q.encode(-1.0)
        assert raw == (65536 - 256)

    def test_signed_literal_positive(self):
        q = Q88()
        lit = q.encode_signed_literal(1.0)
        assert "256" in lit

    def test_signed_literal_negative(self):
        q = Q88()
        lit = q.encode_signed_literal(-1.0)
        # -1.0 in Q8.8 = -256, two's complement = 65280
        assert "65280" in lit or "-256" in lit


class TestCompileLIF:
    def test_basic_lif_generates_verilog(self):
        neuron = from_equations(
            "dv/dt = -(v - E_L)/tau_m + I/C",
            threshold="v > -50",
            reset="v = -65",
            params=dict(E_L=-65, tau_m=10, C=1),
            init=dict(v=-65),
        )
        verilog = compile_to_verilog(neuron, module_name="test_lif")
        assert "module test_lif" in verilog
        assert "endmodule" in verilog
        assert "clk" in verilog
        assert "rst_n" in verilog
        assert "I_t" in verilog
        assert "spike_out" in verilog
        assert "v_out" in verilog
        assert "v_reg" in verilog

    def test_lif_has_threshold_logic(self):
        neuron = from_equations(
            "dv/dt = -(v - E_L)/tau_m + I/C",
            threshold="v > -50",
            reset="v = -65",
            params=dict(E_L=-65, tau_m=10, C=1),
            init=dict(v=-65),
        )
        verilog = compile_to_verilog(neuron)
        assert "spike_out <= 1'b1" in verilog
        assert "spike_out <= 1'b0" in verilog

    def test_lif_has_parameters(self):
        neuron = from_equations(
            "dv/dt = -(v - E_L)/tau_m + I/C",
            threshold="v > -50",
            reset="v = -65",
            params=dict(E_L=-65, tau_m=10, C=1),
            init=dict(v=-65),
        )
        verilog = compile_to_verilog(neuron)
        assert "P_E_L" in verilog
        assert "P_TAU_M" in verilog
        assert "P_C" in verilog


class TestCompileMultiVariable:
    def test_fitzhugh_nagumo(self):
        neuron = EquationNeuron(
            equations={
                "v": "v - v**3 / 3 - w + I",
                "w": "epsilon * (v + a - b * w)",
            },
            parameters={"epsilon": 0.08, "a": 0.7, "b": 0.8},
            state={"v": -1.0, "w": -0.5},
            threshold="v > 1.0",
            reset={"v": "-1.0"},
            dt=0.1,
        )
        verilog = compile_to_verilog(neuron, module_name="fhn_neuron")
        assert "module fhn_neuron" in verilog
        assert "v_reg" in verilog
        assert "w_reg" in verilog
        assert "v_out" in verilog
        assert "w_out" in verilog
        assert "v_next" in verilog
        assert "w_next" in verilog

    def test_izhikevich(self):
        neuron = EquationNeuron(
            equations={
                "v": "0.04 * v**2 + 5 * v + 140 - u + I",
                "u": "a * (b * v - u)",
            },
            parameters={"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
            state={"v": -65.0, "u": -14.0},
            threshold="v > 30",
            reset={"v": "c", "u": "u + d"},
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, module_name="izh_neuron")
        assert "module izh_neuron" in verilog
        assert "v_reg" in verilog
        assert "u_reg" in verilog
        # v**2 should generate a multiply intermediate
        assert "_mul" in verilog


class TestCompileNoThreshold:
    def test_integrator_no_spike(self):
        neuron = EquationNeuron(
            equations={"v": "I"},
            state={"v": 0.0},
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, module_name="integrator")
        assert "module integrator" in verilog
        assert "spike_out <= 1'b0" in verilog
        # No threshold branch
        assert "spike_out <= 1'b1" not in verilog


class TestEquationToFPGA:
    def test_one_liner(self):
        neuron, verilog = equation_to_fpga(
            "dv/dt = -(v - E_L)/tau_m + I/C",
            threshold="v > -50",
            reset="v = -65",
            params=dict(E_L=-65, tau_m=10, C=1),
            init=dict(v=-65),
            module_name="oneliner_lif",
        )
        assert isinstance(neuron, EquationNeuron)
        assert "module oneliner_lif" in verilog
        # Python neuron still works
        spikes = sum(neuron.step(I=30.0) for _ in range(200))
        assert spikes > 0

    def test_multi_eq_one_liner(self):
        neuron, verilog = equation_to_fpga(
            "dv/dt = -(v - v_rest) / tau + I",
            "dw/dt = (v - w) / tau_w",
            params={"v_rest": 0.0, "tau": 10.0, "tau_w": 100.0},
            init={"v": 0.0, "w": 0.0},
            module_name="two_var",
        )
        assert "v_reg" in verilog
        assert "w_reg" in verilog


class TestVerilogSyntax:
    def test_module_structure(self):
        _, verilog = equation_to_fpga(
            "dv/dt = I",
            init={"v": 0.0},
            module_name="syntax_check",
        )
        # Check module/endmodule
        assert verilog.count("endmodule") == 1
        # begin/end balance (endmodule contains "end", so subtract 1)
        assert verilog.count("begin") == verilog.count("end") - 1
        # Has timescale
        assert "`timescale" in verilog
        # Has always block
        assert "always @(posedge clk" in verilog

    def test_auto_generated_comment(self):
        _, verilog = equation_to_fpga(
            "dv/dt = I",
            init={"v": 0.0},
        )
        assert "Auto-generated by SC-NeuroCore" in verilog

    def test_custom_data_width(self):
        neuron = EquationNeuron(
            equations={"v": "I"},
            state={"v": 0.0},
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, data_width=32, fraction=16)
        assert "[31:0]" in verilog
