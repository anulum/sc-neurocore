# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for equation → Verilog compiler

"""Tests for equation_compiler: ODE strings → synthesizable Verilog RTL."""

import pint

from sc_neurocore.neurons.equation_builder import EquationNeuron, from_equations
from sc_neurocore.compiler.equation_compiler import (
    Q88,
    compile_to_verilog,
    equation_to_fpga,
)

UNIT_REGISTRY = pint.UnitRegistry()


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

    def test_unsigned_range_uses_full_width_and_zero_floor(self):
        q = Q88(signed=False)
        # An unsigned UQ8.8 reaches the full 2**16-1 magnitude and floors at zero.
        assert q.max_value == ((1 << 16) - 1) / (1 << 8)
        assert q.min_value == 0.0

    def test_check_range_flags_underflow(self):
        warnings = Q88().check_range(-1000.0, label="v")
        assert any("Underflow" in warning for warning in warnings)


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

    def test_strict_units_lif_compiles_with_named_threshold_and_reset_constants(self):
        neuron = from_equations(
            "dv/dt = (-(v - E_L) + R * I) / tau_m",
            threshold="v > v_threshold",
            reset="v = v_reset",
            params={
                "E_L": -65.0 * UNIT_REGISTRY.millivolt,
                "R": 100e6 * UNIT_REGISTRY.ohm,
                "tau_m": 10.0 * UNIT_REGISTRY.millisecond,
            },
            init={"v": -65.0 * UNIT_REGISTRY.millivolt},
            constants={
                "v_threshold": -50.0 * UNIT_REGISTRY.millivolt,
                "v_reset": -65.0 * UNIT_REGISTRY.millivolt,
            },
            dt=1.0 * UNIT_REGISTRY.millisecond,
            units="strict",
            input_unit=1.0 * UNIT_REGISTRY.nanoampere,
        )
        verilog = compile_to_verilog(neuron, module_name="strict_lif", fraction=16)
        assert "module strict_lif" in verilog
        assert "P_V_THRESHOLD" in verilog
        assert "P_V_RESET" in verilog
        assert "if ((v_next > P_V_THRESHOLD))" in verilog
        assert "v_reg <= P_V_RESET;" in verilog


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

    def test_strict_units_one_liner_export_path(self):
        neuron, verilog = equation_to_fpga(
            "dv/dt = (-(v - E_L) + R * I) / tau_m",
            threshold="v > v_threshold",
            reset="v = v_reset",
            params={
                "E_L": -65.0 * UNIT_REGISTRY.millivolt,
                "R": 100e6 * UNIT_REGISTRY.ohm,
                "tau_m": 10.0 * UNIT_REGISTRY.millisecond,
            },
            init={"v": -65.0 * UNIT_REGISTRY.millivolt},
            constants={
                "v_threshold": -50.0 * UNIT_REGISTRY.millivolt,
                "v_reset": -65.0 * UNIT_REGISTRY.millivolt,
            },
            dt=1.0 * UNIT_REGISTRY.millisecond,
            module_name="strict_oneliner_lif",
            fraction=16,
            units="strict",
            input_unit=1.0 * UNIT_REGISTRY.nanoampere,
        )
        assert "module strict_oneliner_lif" in verilog
        assert "P_V_THRESHOLD" in verilog
        assert str(neuron.get_state()["v"].units) == "millivolt"


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
        assert "32768" in verilog

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

    def test_power_4_compiles(self):
        neuron = EquationNeuron(
            equations={"v": "v**4"},
            state={"v": 1.0},
            dt=0.1,
        )
        verilog = compile_to_verilog(neuron)
        assert "_mul" in verilog  # chained multiplications for power 4

    def test_power_9_raises(self):
        import pytest

        neuron = EquationNeuron(
            equations={"v": "v**9"},
            state={"v": 1.0},
            dt=0.1,
        )
        with pytest.raises(ValueError, match="Only integer powers"):
            compile_to_verilog(neuron)

    def test_sigmoid_in_equation_builder(self):
        """Exercise sigmoid function through equation builder step()."""
        neuron = EquationNeuron(
            equations={"v": "sigmoid(I) - v"},
            state={"v": 0.0},
            dt=0.1,
        )
        neuron.step(I=5.0)
        assert neuron.state["v"] > 0

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

    def test_gte_comparison(self):
        neuron = EquationNeuron(
            equations={"v": "I"},
            state={"v": 0.0},
            threshold="v >= 1",
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, module_name="gte_cmp")
        assert ">=" in verilog

    def test_unsupported_binop_raises(self):
        """Modulo operator is not supported in Verilog emission."""
        import pytest

        neuron = EquationNeuron(
            equations={"v": "v % 2"},
            state={"v": 1.0},
            dt=0.1,
        )
        with pytest.raises(ValueError, match="Unsupported binary op"):
            compile_to_verilog(neuron)

    def test_unsupported_comparison_raises(self):
        import pytest

        neuron = EquationNeuron(
            equations={"v": "I"},
            state={"v": 0.0},
            threshold="v == 0",
            dt=1.0,
        )
        with pytest.raises(ValueError, match="Unsupported comparison"):
            compile_to_verilog(neuron)

    def test_non_name_function_raises(self):
        """Attribute-style calls like obj.method() should raise."""
        import pytest
        import ast

        from sc_neurocore.compiler.equation_compiler import _VerilogExprEmitter, Q88

        q = Q88()
        emitter = _VerilogExprEmitter(set(), {}, q)
        # Construct an ast.Call with ast.Attribute func (not ast.Name)
        node = ast.Call(
            func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="exp", ctx=ast.Load()),
            args=[ast.Constant(value=1.0)],
            keywords=[],
        )
        with pytest.raises(ValueError, match="Only named function calls"):
            emitter.visit_Call(node)

    def test_zero_arg_function_raises(self):
        import pytest
        import ast

        from sc_neurocore.compiler.equation_compiler import _VerilogExprEmitter, Q88

        q = Q88()
        emitter = _VerilogExprEmitter(set(), {}, q)
        node = ast.Call(
            func=ast.Name(id="exp", ctx=ast.Load()),
            args=[],
            keywords=[],
        )
        with pytest.raises(ValueError, match="requires at least 1 argument"):
            emitter.visit_Call(node)

    def test_clip_single_arg_passthrough(self):
        """clip(x) with only 1 arg returns x unchanged."""
        neuron = EquationNeuron(
            equations={"v": "clip(I)"},
            state={"v": 0.0},
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, module_name="clip1")
        assert "module clip1" in verilog

    def test_max_single_arg_passthrough(self):
        """max(x) with only 1 arg returns x unchanged."""
        neuron = EquationNeuron(
            equations={"v": "max(I)"},
            state={"v": 0.0},
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, module_name="max1")
        assert "module max1" in verilog

    def test_trunc_emits_nearest_rounding_pair(self):
        """Nearest rounding adds a half-LSB bias wire before the shift."""
        from sc_neurocore.compiler.equation_compiler import _VerilogExprEmitter, Q88

        emitter = _VerilogExprEmitter({}, {}, Q88(rounding="nearest"))
        trunc_name = emitter._trunc("_wide")
        joined = "\n".join(emitter.intermediates)
        assert "_rnd_half" in joined
        assert trunc_name.startswith("_t")

    def test_trunc_emits_bankers_rounding_guard(self):
        """Banker's rounding adds a tie-detect guard alongside the biased sum."""
        from sc_neurocore.compiler.equation_compiler import _VerilogExprEmitter, Q88

        emitter = _VerilogExprEmitter({}, {}, Q88(rounding="bankers"))
        emitter._trunc("_wide")
        joined = "\n".join(emitter.intermediates)
        assert "_rnd_biased" in joined
        assert "_rnd_guard" in joined

    def test_trunc_emits_stochastic_rounding_lfsr_dither(self):
        """Stochastic rounding dithers the low fraction bits with the LFSR."""
        from sc_neurocore.compiler.equation_compiler import _VerilogExprEmitter, Q88

        emitter = _VerilogExprEmitter({}, {}, Q88(rounding="stochastic"))
        emitter._trunc("_wide")
        joined = "\n".join(emitter.intermediates)
        assert "_rnd_stoch" in joined
        assert "_lfsr" in joined

    def test_trunc_rejects_unknown_rounding_mode(self):
        """An unrecognised rounding mode is rejected by the truncation emitter."""
        import pytest

        from sc_neurocore.compiler.equation_compiler import _VerilogExprEmitter, Q88

        emitter = _VerilogExprEmitter({}, {}, Q88(rounding="dither-supreme"))
        with pytest.raises(ValueError, match="Unknown rounding mode"):
            emitter._trunc("_wide")

    def test_multiply_with_global_pipeline_flag_registers_product(self):
        """The global pipeline flag registers the wide product before truncation."""
        from sc_neurocore.compiler.verilog_expr_emitter import _emit_expr
        from sc_neurocore.compiler.equation_compiler import Q88

        _result, intermediates, _mul, _trunc, pipeline_regs = _emit_expr(
            "v * v", {"v": "v"}, {}, Q88(), pipeline=True
        )
        assert any(reg.startswith("reg signed") and "_r;" in reg for reg in pipeline_regs)
        assert any("_mul0" in line for line in intermediates)

    def test_multiply_with_named_pipeline_point_registers_product(self):
        """A named pipeline insertion point forces product registration."""
        from sc_neurocore.compiler.verilog_expr_emitter import _emit_expr
        from sc_neurocore.compiler.equation_compiler import Q88

        _result, _intermediates, _mul, _trunc, pipeline_regs = _emit_expr(
            "v * v", {"v": "v"}, {}, Q88(), pipeline_points={"_mul0"}
        )
        assert pipeline_regs


class TestCompileCLI:
    """Tests for the sc-neurocore compile CLI command."""

    def test_compile_command_generates_verilog(self, tmp_path):
        from unittest.mock import patch
        from sc_neurocore.cli import main

        out = str(tmp_path / "out")
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
                out,
            ],
        ):
            ret = main()
        assert ret == 0
        import os

        v_path = os.path.join(out, "sc_equation_neuron.v")
        assert os.path.exists(v_path)
        content = (tmp_path / "out" / "sc_equation_neuron.v").read_text()
        assert "module sc_equation_neuron" in content
        assert "endmodule" in content

    def test_compile_with_testbench(self, tmp_path):
        from unittest.mock import patch
        from sc_neurocore.cli import main

        out = str(tmp_path / "tb_out")
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
                out,
                "--module-name",
                "simple",
            ],
        ):
            ret = main()
        assert ret == 0
        import os

        assert os.path.exists(os.path.join(out, "simple.v"))
        assert os.path.exists(os.path.join(out, "tb_simple.v"))

    def test_compile_no_ode_shows_usage(self, capsys):
        from unittest.mock import patch
        from sc_neurocore.cli import main

        with patch("sys.argv", ["sc-neurocore", "compile"]):
            ret = main()
        assert ret == 1
        captured = capsys.readouterr()
        assert "compile requires an ODE string" in captured.out

    def test_compile_with_custom_module_name(self, tmp_path):
        from unittest.mock import patch
        from sc_neurocore.cli import main

        out = str(tmp_path / "custom")
        with patch(
            "sys.argv",
            [
                "sc-neurocore",
                "compile",
                "dv/dt = -v + I",
                "--module-name",
                "my_custom_lif",
                "-o",
                out,
            ],
        ):
            ret = main()
        assert ret == 0
        import os

        assert os.path.exists(os.path.join(out, "my_custom_lif.v"))
        content = (tmp_path / "custom" / "my_custom_lif.v").read_text()
        assert "module my_custom_lif" in content


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


class TestOverflowAndSignednessModes:
    """Codegen branches for the non-default overflow modes and unsigned format."""

    def _unsigned_neuron(self):
        # Non-negative parameters/initial state so the unsigned Q-format can
        # encode every literal.
        return from_equations("dv/dt = -v/tau + I", params=dict(tau=10), init=dict(v=0))

    def _signed_neuron(self):
        return from_equations(
            "dv/dt = -(v - E_L)/tau_m + I/C",
            threshold="v > -50",
            reset="v = -65",
            params=dict(E_L=-65, tau_m=10, C=1),
            init=dict(v=-65),
        )

    def test_unsigned_saturate_emits_unsigned_clamp(self):
        verilog = compile_to_verilog(self._unsigned_neuron(), signed=False, overflow="saturate")
        assert "16'd65535" in verilog  # unsigned saturate ceiling
        assert "16'd0" in verilog  # underflow floor for unsigned
        assert "signed" not in verilog.split("_next")[0].split("\n")[-1]

    def test_wrap_overflow_passes_raw_low_bits(self):
        verilog = compile_to_verilog(self._signed_neuron(), overflow="wrap")
        assert "_next = " in verilog
        assert "[15:0];" in verilog
        assert "OVERFLOW TRAP" not in verilog

    def test_trap_overflow_signed_emits_simulation_assertion(self):
        verilog = compile_to_verilog(self._signed_neuron(), overflow="trap")
        assert "OVERFLOW TRAP" in verilog
        assert "// synthesis translate_off" in verilog
        assert "// synthesis translate_on" in verilog
        assert "$fatal" in verilog

    def test_trap_overflow_unsigned_emits_simulation_assertion(self):
        verilog = compile_to_verilog(self._unsigned_neuron(), signed=False, overflow="trap")
        assert "OVERFLOW TRAP" in verilog
        assert "[16]) " in verilog  # unsigned carry-out overflow check
        assert "$fatal" in verilog
