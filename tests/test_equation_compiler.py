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


class TestTranscendentalFunctions:
    def test_exp_compiles(self):
        neuron = EquationNeuron(
            equations={"v": "exp(-v / tau) + I"},
            parameters={"tau": 10.0},
            state={"v": 0.0},
            dt=0.1,
        )
        verilog = compile_to_verilog(neuron, module_name="exp_neuron")
        assert "module exp_neuron" in verilog
        assert "_exp_lut" in verilog
        assert "case" in verilog

    def test_tanh_compiles(self):
        neuron = EquationNeuron(
            equations={"v": "tanh(I) - v"},
            state={"v": 0.0},
            dt=0.1,
        )
        verilog = compile_to_verilog(neuron, module_name="tanh_neuron")
        assert "_tanh_lut" in verilog

    def test_sigmoid_compiles(self):
        neuron = EquationNeuron(
            equations={"v": "sigmoid(v) * I"},
            state={"v": 0.0},
            dt=0.1,
        )
        verilog = compile_to_verilog(neuron, module_name="sig_neuron")
        assert "_sigmoid_lut" in verilog

    def test_sqrt_compiles(self):
        neuron = EquationNeuron(
            equations={"v": "sqrt(abs(I))"},
            state={"v": 0.0},
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, module_name="sqrt_neuron")
        assert "_sqrt_lut" in verilog

    def test_sin_cos_compile(self):
        neuron = EquationNeuron(
            equations={"v": "sin(v) + cos(I)"},
            state={"v": 0.0},
            dt=0.1,
        )
        verilog = compile_to_verilog(neuron, module_name="trig_neuron")
        assert "_sin_lut" in verilog
        assert "_cos_lut" in verilog

    def test_abs_compiles(self):
        neuron = EquationNeuron(
            equations={"v": "abs(I - v)"},
            state={"v": 0.0},
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, module_name="abs_neuron")
        assert "?" in verilog  # ternary operator from abs

    def test_hodgkin_huxley_gating_compiles(self):
        """The HH alpha_m function uses exp — this was the blocking case."""
        neuron = EquationNeuron(
            equations={"v": "-(v - E_L) / tau + 4 * exp(-(v + 65) / 18)"},
            parameters={"E_L": -65.0, "tau": 10.0},
            state={"v": -65.0},
            dt=0.01,
        )
        verilog = compile_to_verilog(neuron, module_name="hh_gating")
        assert "module hh_gating" in verilog
        assert "_exp_lut" in verilog
        assert "endmodule" in verilog

    def test_log_compiles(self):
        neuron = EquationNeuron(
            equations={"v": "log(abs(v) + 0.01)"},
            state={"v": 1.0},
            dt=0.1,
        )
        verilog = compile_to_verilog(neuron, module_name="log_neuron")
        assert "_log_lut" in verilog

    def test_clip_three_args(self):
        neuron = EquationNeuron(
            equations={"v": "clip(I, -1.0, 1.0)"},
            state={"v": 0.0},
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, module_name="clip_neuron")
        assert "?" in verilog

    def test_max_two_args(self):
        neuron = EquationNeuron(
            equations={"v": "max(I, 0.0)"},
            state={"v": 0.0},
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, module_name="max_neuron")
        assert "?" in verilog

    def test_min_two_args(self):
        neuron = EquationNeuron(
            equations={"v": "min(I, 1.0)"},
            state={"v": 0.0},
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, module_name="min_neuron")
        assert "?" in verilog

    def test_sigmoid_alias_expit(self):
        neuron = EquationNeuron(
            equations={"v": "sigmoid(v)"},
            state={"v": 0.0},
            dt=0.1,
        )
        verilog = compile_to_verilog(neuron, module_name="expit_neuron")
        assert "_sigmoid_lut" in verilog

    def test_nested_transcendentals(self):
        """exp(tanh(v)) — nested function calls."""
        neuron = EquationNeuron(
            equations={"v": "exp(tanh(v)) + I"},
            state={"v": 0.0},
            dt=0.1,
        )
        verilog = compile_to_verilog(neuron, module_name="nested_neuron")
        assert "_exp_lut" in verilog
        assert "_tanh_lut" in verilog

    def test_all_lut_entries_are_integers(self):
        """Verify all LUT helper methods return integer lists."""
        from sc_neurocore.compiler.equation_compiler import _VerilogExprEmitter, Q88

        q = Q88()
        emitter = _VerilogExprEmitter(set(), {}, q)
        for method in [
            emitter._exp_lut_entries,
            emitter._log_lut_entries,
            emitter._sqrt_lut_entries,
            emitter._tanh_lut_entries,
            emitter._sigmoid_lut_entries,
            emitter._sin_lut_entries,
            emitter._cos_lut_entries,
        ]:
            entries = method()
            assert len(entries) == 16
            assert all(isinstance(e, int) for e in entries)

    def test_lut_exp_boundary_values(self):
        """exp(-8) ≈ 0, exp(0) = 256 in Q8.8, exp(7) capped at 32767."""
        from sc_neurocore.compiler.equation_compiler import _VerilogExprEmitter, Q88

        q = Q88()
        emitter = _VerilogExprEmitter(set(), {}, q)
        entries = emitter._exp_lut_entries()
        assert entries[0] < 1  # exp(-8) ≈ 0.000335 → 0 in Q8.8
        assert entries[8] == 256  # exp(0) = 1.0 → 256 in Q8.8
        assert entries[15] == 32767  # exp(7) capped

    def test_lut_tanh_symmetry(self):
        """tanh is odd: tanh(-x) = -tanh(x)."""
        from sc_neurocore.compiler.equation_compiler import _VerilogExprEmitter, Q88

        q = Q88()
        emitter = _VerilogExprEmitter(set(), {}, q)
        entries = emitter._tanh_lut_entries()
        # tanh(-8) ≈ -1.0 → -256, tanh(7) ≈ 1.0 → 256
        assert entries[0] < 0
        assert entries[15] > 0
        # Approximate symmetry around index 8 (x=0)
        assert abs(entries[8]) < 5  # tanh(0) ≈ 0

    def test_lut_sigmoid_range(self):
        """sigmoid output in [0, 1] → [0, 256] in Q8.8."""
        from sc_neurocore.compiler.equation_compiler import _VerilogExprEmitter, Q88

        q = Q88()
        emitter = _VerilogExprEmitter(set(), {}, q)
        entries = emitter._sigmoid_lut_entries()
        assert all(0 <= e <= 256 for e in entries)
        assert entries[0] < 5  # sigmoid(-8) ≈ 0
        assert entries[15] > 250  # sigmoid(7) ≈ 1

    def test_saturating_arithmetic(self):
        _, verilog = equation_to_fpga(
            "dv/dt = I",
            init={"v": 0.0},
            module_name="sat_check",
        )
        assert "v_raw" in verilog
        assert "32767" in verilog
        assert "-32768" in verilog

    def test_unsupported_function_raises(self):
        import pytest

        neuron = EquationNeuron(
            equations={"v": "cosh(v)"},
            state={"v": 0.0},
            dt=1.0,
        )
        with pytest.raises(ValueError, match="cosh"):
            compile_to_verilog(neuron)


class TestTestbenchGenerator:
    def test_generates_testbench(self):
        from sc_neurocore.compiler.equation_compiler import generate_testbench

        neuron = from_equations(
            "dv/dt = -(v - E_L)/tau_m + I/C",
            threshold="v > -50",
            reset="v = -65",
            params=dict(E_L=-65, tau_m=10, C=1),
            init=dict(v=-65),
        )
        tb = generate_testbench(neuron, module_name="test_lif", n_steps=100, input_current=2.0)
        assert "module tb_test_lif" in tb
        assert "endmodule" in tb
        assert "$dumpfile" in tb
        assert "$dumpvars" in tb
        assert "spike_count" in tb
        assert "100" in tb
        assert "v_out" in tb
        assert "uut" in tb

    def test_testbench_multi_variable(self):
        from sc_neurocore.compiler.equation_compiler import generate_testbench

        neuron = EquationNeuron(
            equations={"v": "I - w", "w": "0.01 * v"},
            state={"v": 0.0, "w": 0.0},
            dt=0.1,
        )
        tb = generate_testbench(neuron, module_name="two_var_tb")
        assert "v_out" in tb
        assert "w_out" in tb


class TestEdgeCaseCoverage:
    """Tests for error branches and edge cases."""

    def test_power_4_raises(self):
        import pytest

        neuron = EquationNeuron(
            equations={"v": "v**4"},
            state={"v": 1.0},
            dt=0.1,
        )
        with pytest.raises(ValueError, match="Only integer powers"):
            compile_to_verilog(neuron)

    def test_unary_plus(self):
        neuron = EquationNeuron(
            equations={"v": "+I"},
            state={"v": 0.0},
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, module_name="uadd")
        assert "module uadd" in verilog

    def test_less_than_comparison(self):
        neuron = EquationNeuron(
            equations={"v": "I"},
            state={"v": 0.0},
            threshold="v < 10",
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, module_name="lt_cmp")
        assert "<" in verilog

    def test_less_equal_comparison(self):
        neuron = EquationNeuron(
            equations={"v": "I"},
            state={"v": 0.0},
            threshold="v <= 10",
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, module_name="lte_cmp")
        assert "<=" in verilog

    def test_unknown_name_passthrough(self):
        """Unknown names pass through as-is (for external signals)."""
        neuron = EquationNeuron(
            equations={"v": "external_signal + I"},
            state={"v": 0.0},
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, module_name="ext_sig")
        assert "external_signal" in verilog

    def test_runtime_division(self):
        """Division by non-constant produces raw Verilog division."""
        neuron = EquationNeuron(
            equations={"v": "I / v"},
            state={"v": 1.0},
            dt=0.1,
        )
        verilog = compile_to_verilog(neuron, module_name="rt_div")
        assert "/" in verilog
