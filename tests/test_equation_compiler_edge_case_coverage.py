# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEdgeCaseCoverage from former test_equation_compiler.py

"""Focused suite: TestEdgeCaseCoverage from former test_equation_compiler.py."""

from __future__ import annotations

from tests.equation_compiler_support import *  # noqa: F403

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

    def test_positive_literal_modulo_lowers_python_floor_correction(self) -> None:
        """Positive-literal modulo corrects Verilog's negative remainder."""
        neuron = EquationNeuron(
            equations={"v": "v % 2.0"},
            state={"v": -0.5},
            dt=1.0,
            method="map",
        )

        verilog = compile_to_verilog(neuron, module_name="positive_modulo")

        assert "_mod0_dividend" in verilog
        assert "$signed(_mod0_dividend) % $signed(16'sd512)" in verilog
        assert "(_mod0_remainder < 0) ? (_mod0_remainder + 16'sd512)" in verilog

    @staticmethod
    def test_modulo_rejects_non_positive_or_dynamic_divisors() -> None:
        """Modulo stays narrow: its divisor must be a representable positive literal."""
        import pytest

        for expression in ("v % 0.0", "v % -2.0", "v % period"):
            neuron = EquationNeuron(
                equations={"v": expression},
                parameters={"period": 2.0},
                state={"v": 1.0},
                dt=1.0,
                method="map",
            )
            with pytest.raises(ValueError, match="Modulo divisor"):
                compile_to_verilog(neuron)

    def test_signed_power_of_two_floor_division_lowers_exactly(self) -> None:
        """Floor division keeps Python's negative-floor semantics in Q format."""
        neuron = EquationNeuron(
            equations={"v": "v // 8"},
            state={"v": -10.0},
            dt=1.0,
            method="map",
        )

        verilog = compile_to_verilog(neuron, module_name="signed_floor_division")

        assert "_floordiv0_dividend" in verilog
        assert "$signed(_floordiv0_dividend) >>> 11" in verilog
        assert "_floordiv0_integer <<< 8" in verilog

    def test_floor_division_rejects_dynamic_or_non_power_of_two_divisors(self) -> None:
        """The synthesizable floor subset remains literal and shift-only."""
        import pytest

        for expression in ("v // period", "v // 0", "v // -2", "v // 3", "v // 8.0"):
            neuron = EquationNeuron(
                equations={"v": expression},
                parameters={"period": 8.0},
                state={"v": 1.0},
                dt=1.0,
                method="map",
            )
            with pytest.raises(ValueError, match="Floor divisor"):
                compile_to_verilog(neuron)

    def test_chained_comparison_advances_the_left_operand(self) -> None:
        """``a < b <= c`` must lower as ``a < b && b <= c``."""
        neuron = EquationNeuron(
            equations={"v": "v + I"},
            parameters={"threshold": 1.0},
            state={"v": 0.0},
            threshold="v_prev < threshold <= v_prev + I",
            dt=1.0,
            method="map",
        )

        verilog = compile_to_verilog(neuron, module_name="chained_threshold")

        assert "(v_reg < P_THRESHOLD) && (P_THRESHOLD <= (v_reg + I_t))" in verilog

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
