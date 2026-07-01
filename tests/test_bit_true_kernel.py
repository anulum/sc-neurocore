# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for the bit-true fixed-point kernel generators.

Structural assertions on the emitted C / Rust — the numeric bit-for-bit proof
against the Verilog RTL is in ``tests/test_bit_true_cosim.py``.
"""

from __future__ import annotations

import pytest

from sc_neurocore.compiler.intelligence.bit_true_kernel import (
    _accumulate_bias,
    _ctype,
    _format_tables_c,
    _rtype,
    generate_bittrue_kernel,
    generate_bittrue_kernel_from_neuron,
)
from sc_neurocore.neurons.equation_builder import from_equations


def _lif(dt=1.0):
    return from_equations(
        "dv/dt = -(v - E_L)/tau_m + I/C",
        threshold="v > -50",
        reset="v = -65",
        params=dict(E_L=-65, tau_m=10, C=1),
        init=dict(v=-65),
        dt=dt,
    )


class TestTypeHelpers:
    @pytest.mark.parametrize(
        "dw,expected", [(8, "int8_t"), (16, "int16_t"), (32, "int32_t"), (64, "int64_t")]
    )
    def test_ctype_native(self, dw, expected):
        assert _ctype(dw) == expected

    def test_ctype_non_native_widens(self):
        assert _ctype(24) == "int32_t"
        assert _ctype(48) == "int64_t"

    @pytest.mark.parametrize("dw,expected", [(8, "i8"), (16, "i16"), (32, "i32"), (64, "i64")])
    def test_rtype_native(self, dw, expected):
        assert _rtype(dw) == expected

    def test_rtype_non_native_widens(self):
        assert _rtype(24) == "i32"
        assert _rtype(48) == "i64"

    def test_accumulate_bias_saturate(self):
        assert _accumulate_bias("x", "saturate") == "sat(x)"

    def test_accumulate_bias_wrap(self):
        assert _accumulate_bias("x", "wrap") == "sc_wrap(x, WORD_BITS)"

    def test_format_tables_empty(self):
        assert _format_tables_c({}, 16) == []


class TestSimpleKernelC:
    def test_substrings(self):
        code = generate_bittrue_kernel("sc_lif", {"v": "a + b"})
        for s in ("#include <stdint.h>", "sc_lif_state_t", "sat(", "fxmul("):
            assert s in code

    def test_step_is_not_a_noop(self):
        code = generate_bittrue_kernel("sc_lif", {"v": "a + b"})
        assert "_next_v = sat(" in code
        assert "s->v = _next_v;" in code
        assert "/* update */" not in code  # the old placeholder is gone

    def test_multi_var_struct(self):
        code = generate_bittrue_kernel("sc_izh", {"v": "a * b", "u": "c + d"})
        assert "int16_t v;" in code and "int16_t u;" in code

    def test_free_variables_become_arguments(self):
        code = generate_bittrue_kernel("sc_lif", {"v": "a + b"})
        assert "int16_t a" in code and "int16_t b" in code

    def test_input_current_becomes_argument(self):
        code = generate_bittrue_kernel("sc_lif", {"v": "I - v"})
        assert "int16_t I_t" in code

    def test_transcendental_declares_table(self):
        code = generate_bittrue_kernel("sc_th", {"v": "tanh(v)"})
        assert "static const int16_t _tanh_lut0" in code


class TestSimpleKernelRust:
    def test_substrings(self):
        code = generate_bittrue_kernel("sc_lif", {"v": "a + b"}, language="rust")
        for s in ("pub struct", "fn sat", "clamp"):
            assert s in code

    def test_step_computes(self):
        code = generate_bittrue_kernel("sc_lif", {"v": "a + b"}, language="rust")
        assert "let _next_v" in code and "self.v = _next_v;" in code

    def test_free_variables_become_arguments(self):
        code = generate_bittrue_kernel("sc_lif", {"v": "a + b"}, language="rust")
        assert ", a: i16" in code and ", b: i16" in code


class TestNeuronKernelC:
    def test_reset_and_step(self):
        code = generate_bittrue_kernel_from_neuron(_lif(), "sc_lif")
        assert "sc_lif_reset(sc_lif_state_t *s)" in code
        assert "int sc_lif_step(sc_lif_state_t *s, int16_t I_t)" in code

    def test_bit_identical_claim_present(self):
        code = generate_bittrue_kernel_from_neuron(_lif(), "sc_lif")
        assert "Bit-identical to compile_to_verilog" in code

    def test_threshold_and_spike_sequencing(self):
        code = generate_bittrue_kernel_from_neuron(_lif(), "sc_lif")
        # on spike the output holds the old register value
        assert "s->v_out = s->v;" in code
        assert "int _spk" in code and "return _spk;" in code

    def test_reset_rule_lowered(self):
        code = generate_bittrue_kernel_from_neuron(_lif(), "sc_lif")
        assert "_rst_v = sat(" in code

    def test_no_threshold_branch(self):
        neuron = from_equations("dv/dt = -v + I", init=dict(v=0.0), dt=1.0)
        code = generate_bittrue_kernel_from_neuron(neuron, "sc_leak")
        assert "return 0;" in code and "spike_out = 0;" in code
        assert "_spk" not in code

    def test_multi_var_with_two_resets(self):
        izh = from_equations(
            "dv/dt = 0.04*v**2 + 5*v + 140 - u + I",
            "du/dt = a*(b*v - u)",
            threshold="v > 30",
            reset="v = c; u = u + d",
            params=dict(a=0.02, b=0.2, c=-65, d=8),
            init=dict(v=-65, u=-13),
            dt=1.0,
        )
        code = generate_bittrue_kernel_from_neuron(izh, "sc_izh", data_width=32, fraction=16)
        assert "_rst_v = sat(" in code and "_rst_u = sat(" in code
        assert "int32_t v;" in code and "int32_t u;" in code


class TestNeuronKernelRust:
    def test_reset_and_step(self):
        code = generate_bittrue_kernel_from_neuron(_lif(), "sc_lif", language="rust")
        assert "pub fn reset(&mut self)" in code
        assert "pub fn step(&mut self, I_t: i16) -> i32" in code

    def test_threshold_sequencing(self):
        code = generate_bittrue_kernel_from_neuron(_lif(), "sc_lif", language="rust")
        assert "self.v_out = self.v;" in code and "if _spk != 0" in code

    def test_no_threshold_branch(self):
        neuron = from_equations("dv/dt = -v + I", init=dict(v=0.0), dt=1.0)
        code = generate_bittrue_kernel_from_neuron(neuron, "sc_leak", language="rust")
        assert "return 0;" in code


class TestTranscendentalStatements:
    """Cover the LUT-statement paths (deriv, threshold and table declarations)."""

    def test_simple_rust_lut_and_input(self):
        code = generate_bittrue_kernel("sc_th", {"v": "tanh(v) + I"}, language="rust")
        assert "const _tanh_lut0: [i16;" in code  # rust table declaration
        assert ", I_t: i16" in code  # input argument
        assert "let _tanh_lut0_arg" in code  # LUT statement inside step

    def _transcendental_neuron(self):
        return from_equations(
            "dv/dt = 0.1*(exp(v) - v) + I",
            threshold="tanh(v) > 0.5",
            reset="v = 0",
            init=dict(v=0.0),
            dt=0.5,
        )

    def test_neuron_c_deriv_and_threshold_statements(self):
        code = generate_bittrue_kernel_from_neuron(self._transcendental_neuron(), "sc_tr")
        assert "_exp_lut0_arg" in code  # derivative LUT statement (line 534)
        assert "_tanh_lut" in code  # threshold LUT statement (line 557)

    def test_neuron_rust_deriv_and_threshold_statements(self):
        code = generate_bittrue_kernel_from_neuron(
            self._transcendental_neuron(), "sc_tr", language="rust"
        )
        assert "let _exp_lut0_arg" in code  # derivative LUT statement (line 609)
        assert "_tanh_lut" in code  # threshold LUT statement (line 634)


class TestModesAndValidation:
    def test_nearest_rounding_adds_half(self):
        code = generate_bittrue_kernel_from_neuron(_lif(), "sc_lif", rounding="nearest")
        assert "(1 << (FRAC_BITS - 1))" in code

    def test_wrap_overflow_uses_sc_wrap(self):
        code = generate_bittrue_kernel_from_neuron(_lif(), "sc_lif", overflow="wrap")
        assert "sc_wrap(((int64_t)s->v)" in code

    def test_nearest_rounding_rust(self):
        code = generate_bittrue_kernel_from_neuron(
            _lif(), "sc_lif", rounding="nearest", language="rust"
        )
        assert "(1 << (FRAC_BITS - 1))" in code

    def test_bad_language_simple(self):
        with pytest.raises(ValueError, match="language must be"):
            generate_bittrue_kernel("m", {"v": "v"}, language="go")

    def test_bad_language_neuron(self):
        with pytest.raises(ValueError, match="language must be"):
            generate_bittrue_kernel_from_neuron(_lif(), "m", language="go")

    def test_bankers_rounding_rejected(self):
        with pytest.raises(ValueError, match="rounding"):
            generate_bittrue_kernel_from_neuron(_lif(), "m", rounding="bankers")

    def test_stochastic_rounding_rejected(self):
        with pytest.raises(ValueError, match="stochastic"):
            generate_bittrue_kernel_from_neuron(_lif(), "m", rounding="stochastic")

    def test_trap_overflow_rejected(self):
        with pytest.raises(ValueError, match="trap"):
            generate_bittrue_kernel_from_neuron(_lif(), "m", overflow="trap")

    def test_unsigned_rejected(self):
        with pytest.raises(ValueError, match="signed=True"):
            generate_bittrue_kernel_from_neuron(_lif(), "m", signed=False)

    def test_dt_underflow_rejected(self):
        # dt=0.001 underflows Q8.8 (resolution 1/256 ≈ 0.0039)
        neuron = from_equations("dv/dt = -v + I", init=dict(v=0.0), dt=0.001)
        with pytest.raises(ValueError, match="underflows"):
            generate_bittrue_kernel_from_neuron(neuron, "m", data_width=16, fraction=8)
