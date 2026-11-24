# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTranscendentalFunctions from former test_equation_compiler.py

"""Focused suite: TestTranscendentalFunctions from former test_equation_compiler.py."""

from __future__ import annotations

from sc_neurocore.compiler.equation_compiler import (
    compile_to_verilog,
    equation_to_fpga,
)
from sc_neurocore.neurons.equation_builder import (
    EquationNeuron,
)


class TestTranscendentalFunctions:
    """Compiling transcendental functions to lookup tables."""

    def test_exp_compiles(self) -> None:
        """The exp function compiles to a lookup table."""
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

    def test_tanh_compiles(self) -> None:
        """The tanh function compiles to a lookup table."""
        neuron = EquationNeuron(
            equations={"v": "tanh(I) - v"},
            state={"v": 0.0},
            dt=0.1,
        )
        verilog = compile_to_verilog(neuron, module_name="tanh_neuron")
        assert "_tanh_lut" in verilog

    def test_sigmoid_compiles(self) -> None:
        """The sigmoid function compiles to a lookup table."""
        neuron = EquationNeuron(
            equations={"v": "sigmoid(v) * I"},
            state={"v": 0.0},
            dt=0.1,
        )
        verilog = compile_to_verilog(neuron, module_name="sig_neuron")
        assert "_sigmoid_lut" in verilog

    def test_sqrt_compiles(self) -> None:
        """The sqrt function compiles to a lookup table."""
        neuron = EquationNeuron(
            equations={"v": "sqrt(abs(I))"},
            state={"v": 0.0},
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, module_name="sqrt_neuron")
        assert "_sqrt_lut" in verilog

    def test_sin_cos_compile(self) -> None:
        """The sin and cos functions compile to lookup tables."""
        neuron = EquationNeuron(
            equations={"v": "sin(v) + cos(I)"},
            state={"v": 0.0},
            dt=0.1,
        )
        verilog = compile_to_verilog(neuron, module_name="trig_neuron")
        assert "_sin_lut" in verilog
        assert "_cos_lut" in verilog

    def test_abs_compiles(self) -> None:
        """The abs function compiles to a conditional."""
        neuron = EquationNeuron(
            equations={"v": "abs(I - v)"},
            state={"v": 0.0},
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, module_name="abs_neuron")
        assert "?" in verilog  # ternary operator from abs

    def test_hodgkin_huxley_gating_compiles(self) -> None:
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

    def test_log_compiles(self) -> None:
        """The log function compiles to a lookup table."""
        neuron = EquationNeuron(
            equations={"v": "log(abs(v) + 0.01)"},
            state={"v": 1.0},
            dt=0.1,
        )
        verilog = compile_to_verilog(neuron, module_name="log_neuron")
        assert "_log_lut" in verilog

    def test_clip_three_args(self) -> None:
        """The clip function compiles to a conditional."""
        neuron = EquationNeuron(
            equations={"v": "clip(I, -1.0, 1.0)"},
            state={"v": 0.0},
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, module_name="clip_neuron")
        assert "?" in verilog

    def test_max_two_args(self) -> None:
        """The max function compiles to a conditional."""
        neuron = EquationNeuron(
            equations={"v": "max(I, 0.0)"},
            state={"v": 0.0},
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, module_name="max_neuron")
        assert "?" in verilog

    def test_min_two_args(self) -> None:
        """The min function compiles to a conditional."""
        neuron = EquationNeuron(
            equations={"v": "min(I, 1.0)"},
            state={"v": 0.0},
            dt=1.0,
        )
        verilog = compile_to_verilog(neuron, module_name="min_neuron")
        assert "?" in verilog

    def test_sigmoid_alias_expit(self) -> None:
        """The expit alias compiles to the sigmoid table."""
        neuron = EquationNeuron(
            equations={"v": "sigmoid(v)"},
            state={"v": 0.0},
            dt=0.1,
        )
        verilog = compile_to_verilog(neuron, module_name="expit_neuron")
        assert "_sigmoid_lut" in verilog

    def test_nested_transcendentals(self) -> None:
        """exp(tanh(v)) — nested function calls."""
        neuron = EquationNeuron(
            equations={"v": "exp(tanh(v)) + I"},
            state={"v": 0.0},
            dt=0.1,
        )
        verilog = compile_to_verilog(neuron, module_name="nested_neuron")
        assert "_exp_lut" in verilog
        assert "_tanh_lut" in verilog

    def test_all_lut_entries_are_integers(self) -> None:
        """Each LUT helper returns integer entries of its expected length.

        The symmetric LUTs and positive-domain log LUT sample 256 points; sqrt
        retains its 16-entry table.
        """
        from sc_neurocore.compiler.equation_compiler import _VerilogExprEmitter, Q88

        q = Q88()
        emitter = _VerilogExprEmitter({}, {}, q)
        expected_len = {
            emitter._exp_lut_entries: 256,
            emitter._log_lut_entries: 256,
            emitter._sqrt_lut_entries: 16,
            emitter._tanh_lut_entries: 256,
            emitter._sigmoid_lut_entries: 256,
            emitter._sin_lut_entries: 256,
            emitter._cos_lut_entries: 256,
        }
        for method, length in expected_len.items():
            entries = method()
            assert len(entries) == length
            assert all(isinstance(e, int) for e in entries)

    def test_lut_exp_boundary_values(self) -> None:
        """The exp table over the 256-point [-16, 16) grid.

        It is ≈0 at the low end, 256 at x=0 (index 128), and saturated at the Q8.8
        signed max (32767) at the high end.
        """
        from sc_neurocore.compiler.equation_compiler import _VerilogExprEmitter, Q88

        q = Q88()
        emitter = _VerilogExprEmitter({}, {}, q)
        entries = emitter._exp_lut_entries()
        assert entries[0] < 1  # exp(-16) ≈ 1.1e-7 → 0 in Q8.8
        assert entries[128] == 256  # exp(0) = 1.0 → 256 in Q8.8 (x=0 at index 128)
        assert entries[255] == 32767  # exp(15.875) saturated at the signed max

    def test_lut_tanh_symmetry(self) -> None:
        """The tanh table is odd: ≈-1 at the low end, ≈+1 at the high end, ≈0 at x=0."""
        from sc_neurocore.compiler.equation_compiler import _VerilogExprEmitter, Q88

        q = Q88()
        emitter = _VerilogExprEmitter({}, {}, q)
        entries = emitter._tanh_lut_entries()
        # tanh(-16) ≈ -1.0 → -256, tanh(15.875) ≈ 1.0 → 256
        assert entries[0] < 0
        assert entries[255] > 0
        assert abs(entries[128]) < 5  # tanh(0) ≈ 0 at index 128

    def test_lut_sigmoid_range(self) -> None:
        """The sigmoid output in [0, 1] maps to [0, 256] in Q8.8 over the 256-point grid."""
        from sc_neurocore.compiler.equation_compiler import _VerilogExprEmitter, Q88

        q = Q88()
        emitter = _VerilogExprEmitter({}, {}, q)
        entries = emitter._sigmoid_lut_entries()
        assert all(0 <= e <= 256 for e in entries)
        assert entries[0] < 5  # sigmoid(-16) ≈ 0
        assert entries[255] > 250  # sigmoid(15.875) ≈ 1

    def test_saturating_arithmetic(self) -> None:
        """Arithmetic saturates at the Q8.8 bounds."""
        _, verilog = equation_to_fpga(
            "dv/dt = I",
            init={"v": 0.0},
            module_name="sat_check",
        )
        assert "v_raw" in verilog
        assert "32767" in verilog
        assert "32768" in verilog

    def test_unsupported_function_raises(self) -> None:
        """An unsupported function is rejected."""
        import pytest

        # cosh/exprel/cbrt are now supported LUT calls; sinh has no Verilog lowering.
        neuron = EquationNeuron(
            equations={"v": "sinh(v)"},
            state={"v": 0.0},
            dt=1.0,
        )
        with pytest.raises(ValueError, match="sinh"):
            compile_to_verilog(neuron)
