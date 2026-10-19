# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — measured look-up-table error against the Python model

"""Tests of :mod:`sc_neurocore.compiler.lut_accuracy`.

``lut_read`` is checked against the generated RTL itself (Icarus Verilog, skipped
when absent), so the measured errors describe the hardware, not a second model of
it. The measured maxima are checked against values worked out independently.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.compiler import expr_lut_tables as tables
from sc_neurocore.compiler.hardware_numeric_contract import hardware_numeric_contract
from sc_neurocore.compiler.lut_accuracy import (
    EXHAUSTIVE_MAX_WIDTH,
    GEOMETRY,
    lut_read,
    measure,
    table_functions,
)
from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.bit_true_cosim_support import HAS_COSIM, _q_input, _verilog_trace

_EXPRESSION = {"cbrt": "I ** (1.0 / 3.0)", "sigmoid": "sigmoid(I)"}
_POSITIVE = {"log", "sqrt", "cbrt"}


class TestVocabulary:
    """The measured set is the set of tables the lowering builds."""

    def test_every_table_function_is_measured(self) -> None:
        """Every table-lowered function of the vocabulary, plus the two root powers."""
        lowered = {"exp", "log", "sqrt", "tanh", "cosh", "exprel", "sigmoid", "sin", "cos"}
        assert lowered <= tables.SUPPORTED_FUNCTIONS
        assert set(table_functions()) == lowered | {"cbrt"}

    def test_unknown_function_and_bad_format_are_refused(self) -> None:
        """No measurement is invented for something the lowering does not build."""
        with pytest.raises(ValueError, match="no lut-v1 table"):
            measure("erf", 16, 8)
        with pytest.raises(ValueError, match="invalid format"):
            measure("exp", 16, 16)


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
class TestLutReadIsTheRtl:
    """``lut_read`` selects the entry the generated RTL selects."""

    @pytest.mark.parametrize(("dw", "fr"), [(16, 8), (16, 12), (32, 16)])
    @pytest.mark.parametrize("function", sorted(GEOMETRY))
    def test_against_simulation(self, function: str, dw: int, fr: int) -> None:
        """Inputs on, just below and between sample points read the same entry in RTL and here."""
        geometry = GEOMETRY[function]
        expression = _EXPRESSION.get(function, f"{function}(I)")
        neuron = EquationNeuron(equations={"v": expression}, state={"v": 0.0}, dt=1.0, method="map")
        point = geometry.lo + 10 * geometry.step
        inputs = [point, point - 2.0**-fr, point + geometry.step / 2]
        if function not in _POSITIVE:
            inputs.append(-0.75)
        for x in inputs:
            code = _q_input(x, dw, fr)
            expected = int(lut_read(function, np.array([code]), dw, fr)[0])
            with tempfile.TemporaryDirectory() as d:
                rtl = _verilog_trace(neuron, f"acc_{function}", x, 2, dw, fr, Path(d))
            assert int(rtl[-1][1]) == expected, f"{function}({x}) at Q{dw - fr}.{fr}"


class TestMeasuredValues:
    """Measured maxima equal values worked out independently."""

    def test_exp_q8_8_worst_case_by_hand(self) -> None:
        """At Q8.8 the worst input reads the entry of the sample point below it."""
        result = measure("exp", 16, 8)
        assert result.exhaustive and result.codes_evaluated == 1 << 16
        x = result.max_error_at
        sample = -16.0 + math.floor((x + 16.0) * 8) / 8
        entry = round(math.exp(sample) * 256) / 256
        assert result.max_error == pytest.approx(abs(entry - math.exp(x)))
        assert result.max_error > 13.0  # the table is coarse at steep exp; stated, not hidden

    def test_log_evaluates_only_its_domain(self) -> None:
        """``log`` is measured on positive inputs only: codes 1 .. 2**15 - 1 at Q8.8."""
        result = measure("log", 16, 8)
        assert result.codes_evaluated == (1 << 15) - 1
        assert result.input_min == 1 / 256

    def test_wide_format_is_sampled_and_still_finds_the_cell_maximum(self) -> None:
        """At Q16.16 the per-cell maximum of increasing ``exp`` lies at each cell's last code."""
        result = measure("exp", 32, 16)
        assert not result.exhaustive and EXHAUSTIVE_MAX_WIDTH < 32
        entries = tables.exp_lut_entries(32, 16)
        cap = ((1 << 31) - 1) / 65536
        ends = [-16.0 + (k + 1) * 0.125 - 2.0**-16 for k in range(256)]
        by_hand = max(abs(entries[k] / 65536 - min(math.exp(x), cap)) for k, x in enumerate(ends))
        assert result.max_error_in_table == pytest.approx(by_hand)

    def test_sigmoid_error_is_bounded_by_its_slope(self) -> None:
        """The steepest slope of the logistic function is 1/4, so a 1/8 step costs at most 1/32."""
        for dw, fr in [(16, 8), (16, 12), (32, 16)]:
            assert measure("sigmoid", dw, fr).max_error <= 1 / 32 + 2.0 ** -(fr + 1)

    def test_public_projection_states_units_and_reference(self) -> None:
        """The JSON states errors in real units and in least significant bits."""
        public = measure("tanh", 16, 8).to_public_dict()
        assert public["q_format"] == "Q8.8" and public["lowering"] == "lut-v1"
        result = measure("tanh", 16, 8)
        assert public["max_error_lsb"] == pytest.approx(result.max_error * 256)
        assert "clipped to the word's range" in str(public["reference"])


class TestV2Measurement:
    """``lowering="v2"`` measures the function units the same way."""

    def test_v2_functions_include_sinh(self) -> None:
        """The v2 vocabulary is the eleven units; lut-v1 has no sinh."""
        assert "sinh" in table_functions("v2") and "sinh" not in table_functions()
        with pytest.raises(ValueError, match="unknown lowering"):
            table_functions("v3")

    def test_q8_8_units_are_correctly_rounded_except_two_term_ones(self) -> None:
        """Every Q8.8 code: 0.5 LSB; cosh and sinh within 0.63, tan within 0.52 LSB."""
        bounds = {"cosh": 0.63, "sinh": 0.63, "tan": 0.52}
        for function in table_functions("v2"):
            result = measure(function, 16, 8, "v2")
            assert result.exhaustive and result.lowering == "v2"
            assert result.max_error / result.resolution <= (bounds.get(function, 0.501))

    def test_v2_improves_every_shared_function_at_q16_16(self) -> None:
        """Inside its domain v2 is at least 1000 times closer than lut-v1 for exp, sin and sigmoid."""
        for function in ("exp", "sin", "sigmoid"):
            v1 = measure(function, 32, 16).max_error_in_table
            v2 = measure(function, 32, 16, "v2").max_error
            assert v2 * 1000 < v1

    def test_unknown_v2_function_is_refused(self) -> None:
        """No measurement is invented for a unit that does not exist."""
        with pytest.raises(ValueError, match="no v2 unit"):
            measure("gamma", 16, 8, "v2")


class TestContract:
    """The hardware numeric contract carries the measurement for its own format."""

    def test_table_carries_accuracy_at_the_contract_format(self) -> None:
        """A neuron using ``tanh`` at Q4.12 reports the Q4.12 measurement."""
        neuron = EquationNeuron(equations={"v": "tanh(v) + I"}, method="map", dt=1.0)
        contract = hardware_numeric_contract(neuron, QFormat.from_string("Q4.12"))
        (table,) = contract.lookup_tables
        assert table.accuracy == measure("tanh", 16, 12)
        assert table.to_public_dict()["accuracy"] == measure("tanh", 16, 12).to_public_dict()

    def test_v2_contract_reports_units_not_tables(self) -> None:
        """With function_lowering v2 the contract lists the units with their v2 measurement."""
        neuron = EquationNeuron(
            equations={"v": "tanh(v) + v ** (1.0 / 3.0) + I"}, method="map", dt=1.0
        )
        contract = hardware_numeric_contract(
            neuron, QFormat.from_string("Q8.8"), function_lowering="v2"
        )
        assert contract.lookup_tables == ()
        assert [unit.function for unit in contract.function_units] == ["cbrt", "tanh"]
        assert contract.function_units[1].accuracy == measure("tanh", 16, 8, "v2")
        public = contract.to_public_dict()
        units = public["function_units"]
        assert public["function_lowering"] == "v2" and isinstance(units, list) and len(units) == 2

    def test_unknown_lowering_is_refused_by_the_contract(self) -> None:
        """The contract refuses a lowering the compiler does not have."""
        neuron = EquationNeuron(equations={"v": "tanh(v)"}, method="map", dt=1.0)
        with pytest.raises(ValueError, match="function_lowering must be"):
            hardware_numeric_contract(neuron, QFormat.from_string("Q8.8"), function_lowering="v3")


@pytest.mark.parametrize(("function", "wide"), [("cos", 32), ("cos", 62), ("sin", 32)])
def test_a_wide_coarse_format_samples_every_code_inside_the_table(function: str, wide: int) -> None:
    """RIGOR review 269 (NEURO-E4B-LUT-SAMPLING-01): at fraction 0 the cell starts collapsed.

    A 1/8 step rounds to 0 codes at fraction 0, so the sampled 32/0 and 62/0 cos
    reported 0.042 at -16 although the 32 integer codes in the table reach 0.4597 at
    -1. The wide result stays sampled (stated as such) and now equals the exhaustive
    16/0 control, whose in-table codes are the same.
    """
    sampled = measure(function, wide, 0)
    exhaustive = measure(function, 16, 0)
    assert not sampled.exhaustive and exhaustive.exhaustive
    assert sampled.max_error_in_table == exhaustive.max_error_in_table
    assert sampled.max_error_in_table_at == exhaustive.max_error_in_table_at
