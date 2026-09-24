# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware numeric contract against the generated RTL

"""The contract states what the generated RTL holds, checked against that RTL.

Every neuron here is compiled by the production equation compiler; the values
the contract reports are compared with the words and look-up tables the
emitted Verilog actually contains.
"""

from __future__ import annotations

import math
import re

import pytest

from sc_neurocore.compiler.hardware_numeric_contract import (
    HARDWARE_NUMERIC_CONTRACT_SCHEMA_VERSION,
    MIRROR_EVIDENCE,
    NOT_STATED,
    HardwareNumericContract,
    hardware_numeric_contract,
)
from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron

Q88 = QFormat.from_string("Q8.8")
Q1616 = QFormat.from_string("Q16.16")

_PARAMETER = re.compile(r"parameter signed \[(\d+):0\] P_(\w+) = \d+'sd(\d+)")
_LUT = re.compile(
    r"// _(\w+)_lut lookup table \((\d+) entries over \[([-\d.e]+), ([-\d.e]+)\), step ([\d.e-]+)\)"
)


def _rtl(neuron: EquationNeuron, q_format: QFormat) -> str:
    return compile_to_verilog(
        neuron, data_width=q_format.total_bits, fraction=q_format.fraction_bits
    )


def _signed(word: int, width: int) -> int:
    """Read a Verilog sized literal the way a ``signed [width-1:0]`` net holds it."""
    word &= (1 << width) - 1
    return word - (1 << width) if word >= 1 << (width - 1) else word


def _rtl_parameters(verilog: str, fraction: int) -> dict[str, float]:
    return {
        name: _signed(int(word), int(msb) + 1) / (1 << fraction)
        for msb, name, word in _PARAMETER.findall(verilog)
    }


def _rtl_tables(verilog: str) -> set[tuple[str, int, float, float, float]]:
    return {
        (name, int(n), float(lo), float(hi), float(step))
        for name, n, lo, hi, step in _LUT.findall(verilog)
    }


def _tables(contract: HardwareNumericContract) -> set[tuple[str, int, float, float, float]]:
    return {
        (table.function, table.entries, table.domain_min, table.domain_max, table.step)
        for table in contract.lookup_tables
    }


def _lif() -> EquationNeuron:
    return EquationNeuron(
        equations={"v": "(-v + a * I - g * v) / tau + b"},
        parameters={"a": 0.1, "tau": 200.0, "g": 0.001, "b": 0.5},
        state={"v": -65.0},
        threshold="v >= 30",
        reset={"v": "-65.0"},
        dt=0.1,
    )


class TestEncodedValues:
    def test_each_value_holds_in_the_contract_what_it_holds_in_the_rtl(self) -> None:
        neuron = _lif()
        contract = hardware_numeric_contract(neuron, Q88)
        named = {(q.kind, q.name): q for q in contract.quantities}

        rtl = _rtl_parameters(_rtl(neuron, Q88), fraction=8)
        for name in ("a", "tau", "g", "b"):
            assert named[("parameter", name)].rtl_value == rtl[name.upper()]

        assert [
            (q.status, q.blocking)
            for q in (named[("parameter", n)] for n in ["a", "tau", "g", "b"])
        ] == [
            ("rounded", False),
            ("out_of_range", True),
            ("underflows_to_zero", True),
            ("exact", False),
        ]
        assert named[("parameter", "tau")].rtl_value == -56.0
        assert named[("parameter", "a")].relative_error == pytest.approx(abs(26 / 256 - 0.1) / 0.1)
        assert named[("initial_state", "v")].status == "exact"
        assert named[("time_step", "dt")].rtl_value == 26 / 256
        assert contract.representable is False
        assert contract.refusal() == (
            "Q8.8 cannot hold this neuron (range [-128.0, 127.99609375], resolution "
            "0.00390625): parameter tau=200.0 becomes -56.0; parameter g=0.001 becomes 0.0"
        )

    def test_the_same_neuron_fits_a_wider_format(self) -> None:
        contract = hardware_numeric_contract(_lif(), Q1616)
        assert contract.representable is True
        assert contract.refusal() == ""
        assert contract.blocking == ()
        rtl = _rtl_parameters(_rtl(_lif(), Q1616), fraction=16)
        assert {q.name: q.rtl_value for q in contract.quantities if q.kind == "parameter"} == {
            name.lower(): value for name, value in rtl.items()
        }

    def test_a_zero_value_has_no_relative_error(self) -> None:
        contract = hardware_numeric_contract(
            EquationNeuron(equations={"v": "-v"}, state={"v": 0.0}), Q88
        )
        (initial,) = [q for q in contract.quantities if q.kind == "initial_state"]
        assert (initial.status, initial.relative_error) == ("exact", None)
        assert initial.to_public_dict() == {
            "kind": "initial_state",
            "name": "v",
            "value": 0.0,
            "rtl_value": 0.0,
            "status": "exact",
            "relative_error": None,
            "blocking": False,
        }

    def test_constants_are_encoded_like_parameters_and_a_zero_step_is_not(self) -> None:
        neuron = EquationNeuron(
            equations={"v": "v * k"}, constants={"k": 300.0}, method="map", dt=0.0
        )
        contract = hardware_numeric_contract(neuron, Q88)
        (constant,) = [q for q in contract.quantities if q.kind == "constant"]
        assert constant.rtl_value == _rtl_parameters(_rtl(neuron, Q88), fraction=8)["K"] == 44.0
        assert (constant.status, constant.blocking) == ("out_of_range", True)
        assert not [q for q in contract.quantities if q.kind == "time_step"]

    @pytest.mark.parametrize("value", [math.inf, math.nan])
    def test_a_non_finite_value_is_refused(self, value: float) -> None:
        neuron = EquationNeuron(equations={"v": "-v * a"}, parameters={"a": value})
        with pytest.raises(ValueError, match="parameter a is not finite"):
            hardware_numeric_contract(neuron, Q88)


class TestLiterals:
    def test_literals_are_read_as_the_emitter_encodes_them(self) -> None:
        """Exponents and floor divisors are not encoded; divisors become reciprocals."""
        neuron = EquationNeuron(
            equations={
                "v": "v**2 + v // 4 + v % 2.5 + v / 1000 + 0.001 * v + 0.001 * v + (v if True else 0)"
            },
            method="map",
            dt=0.5,
        )
        contract = hardware_numeric_contract(neuron, Q88)
        literals = [
            (q.kind, q.value, q.status, q.blocking)
            for q in contract.quantities
            if q.kind not in {"initial_state", "time_step"}
        ]
        assert literals == [
            ("modulo_period", 2.5, "exact", False),
            ("literal_divisor", 0.001, "underflows_to_zero", True),
            ("literal", 0.001, "underflows_to_zero", False),
            ("literal", 0.0, "exact", False),
        ]
        # The reciprocal word the RTL multiplies by is the one the contract names.
        assert re.search(r"= \S+ \* 16'sd0;", _rtl(neuron, Q88))
        assert contract.refusal().endswith("literal divisor equation v=0.001 becomes 0.0")

    def test_a_literal_below_the_resolution_is_reported_not_blocking(self) -> None:
        neuron = EquationNeuron(equations={"v": "v - 1e-12"}, method="map", dt=1.0)
        contract = hardware_numeric_contract(neuron, Q1616)
        (guard,) = [q for q in contract.quantities if q.kind == "literal"]
        assert (guard.status, guard.blocking) == ("underflows_to_zero", False)
        assert contract.representable is True

    def test_division_by_the_literal_zero_is_refused(self) -> None:
        with pytest.raises(ValueError, match="divides by the literal 0"):
            hardware_numeric_contract(EquationNeuron(equations={"v": "v / 0"}), Q88)


class TestIntegrators:
    def test_rk4_states_its_half_and_sixth_step_scales(self) -> None:
        neuron = EquationNeuron(equations={"v": "-v"}, method="rk4", dt=0.01)
        narrow = hardware_numeric_contract(neuron, Q88)
        steps = {q.name: (q.status, q.blocking) for q in narrow.quantities if q.kind == "time_step"}
        assert steps == {
            "dt": ("rounded", False),
            "dt/2": ("rounded", False),
            "dt/6": ("underflows_to_zero", True),
        }
        assert hardware_numeric_contract(neuron, Q1616).representable is True

    def test_exponential_euler_states_its_jacobian_and_exprel_table(self) -> None:
        neuron = EquationNeuron(equations={"v": "(-v + I) / 10.0"}, method="exp_euler", dt=0.5)
        contract = hardware_numeric_contract(neuron, Q88)
        assert ("literal", "jacobian v") in {(q.kind, q.name) for q in contract.quantities}
        assert _tables(contract) == _rtl_tables(_rtl(neuron, Q88))
        assert [table.function for table in contract.lookup_tables] == ["exprel"]


class TestLookupTables:
    def test_every_table_the_rtl_builds_is_stated_with_its_domain(self) -> None:
        neuron = EquationNeuron(
            equations={
                "v": "exp(v) + log(v) + sqrt(v) + tanh(v) + cosh(v) + expit(v)",
                "w": "sin(w) + cos(w) + exprel(w) + w**(1/3) + w**0.5 + abs(w)",
            },
            method="map",
            dt=1.0,
        )
        contract = hardware_numeric_contract(neuron, Q88)
        assert _tables(contract) == _rtl_tables(_rtl(neuron, Q88))
        by_function = {table.function: table for table in contract.lookup_tables}
        assert (by_function["log"].domain_min, by_function["log"].step) == (1 / 256, 1 / 32)
        assert (by_function["sqrt"].domain_max, by_function["sqrt"].entries) == (8.0, 16)
        assert by_function["cbrt"].to_public_dict() == {
            "function": "cbrt",
            "domain_min": -16.0,
            "domain_max": 16.0,
            "step": 0.125,
            "entries": 256,
            "outside_domain": "clamped to the first or last entry",
        }

    def test_stochastic_spike_expressions_are_read(self) -> None:
        escape = EquationNeuron(
            equations={"v": "-v + I"},
            detection="escape_rate",
            rate_expression="0.5 * exp(v)",
            reset={"v": "0.0"},
        )
        poisson = EquationNeuron(
            equations={"v": "-v + I"}, detection="poisson", probability_expression="sigmoid(v)"
        )
        escape_contract = hardware_numeric_contract(escape, Q88)
        assert ("literal", "escape rate", 0.5) in {
            (q.kind, q.name, q.value) for q in escape_contract.quantities
        }
        assert [t.function for t in escape_contract.lookup_tables] == ["exp"]
        assert [t.function for t in hardware_numeric_contract(poisson, Q88).lookup_tables] == [
            "sigmoid"
        ]


class TestMirrorAndProjection:
    def test_an_euler_neuron_has_a_bit_true_mirror_and_its_arithmetic(self) -> None:
        contract = hardware_numeric_contract(_lif(), Q1616)
        public = contract.to_public_dict()
        mirror = public["bit_true_mirror"]
        assert isinstance(mirror, dict)
        assert mirror["available"] is True and mirror["refusal"] == ""
        assert mirror["evidence"] == MIRROR_EVIDENCE
        assert mirror["arithmetic"]["q_format"] == "Q16.16"
        assert mirror["arithmetic"]["method"] == "euler"
        assert public["schema_version"] == HARDWARE_NUMERIC_CONTRACT_SCHEMA_VERSION
        assert (public["min_value"], public["max_value"]) == (-32768.0, 32767.9999847412109375)
        assert public["resolution"] == 1 / 65536
        assert (public["method"], public["overflow"], public["rounding"]) == (
            "euler",
            "saturate",
            "truncate",
        )
        assert public["not_stated"] == list(NOT_STATED)
        assert public["timing"] == {
            "cycles_per_step": 1,
            "pipeline_latency_cycles": 0,
            "basis": (
                "compiled without pipeline stages: every state register and the spike "
                "output take their next value on each rising clock edge"
            ),
        }
        # The RTL the timing statement describes has no pipeline latency port.
        assert "latency" not in _rtl(_lif(), Q1616)

    def test_a_neuron_without_a_mirror_says_why(self) -> None:
        contract = hardware_numeric_contract(
            EquationNeuron(equations={"v": "-v"}, method="rk4", dt=0.5), Q88
        )
        mirror = contract.to_public_dict()["bit_true_mirror"]
        assert contract.arithmetic is None
        assert mirror == {
            "available": False,
            "refusal": (
                "bit-true neuron kernel currently supports method='euler' or "
                "method='map', got 'rk4'"
            ),
            "evidence": "",
            "arithmetic": None,
        }


class TestDerivedDivisors:
    """A divisor built from parameters is evaluated as the datapath computes it."""

    @pytest.mark.parametrize(
        ("divisor", "zero_divisors"),
        [
            (
                "(1.0 / a - 1.0 / b) * (1.0 / a - 1.0 / b)",
                ["(1.0 / a - 1.0 / b) * (1.0 / a - 1.0 / b)"],
            ),
            ("a - a", ["a - a"]),
            ("+a - a", ["+a - a"]),
            ("a ** 2 - a ** 2", ["a ** 2 - a ** 2"]),
            ("a / 4.0 - b / 2.0", ["a / 4.0 - b / 2.0"]),
            ("a + x", []),
            ("x + a", []),
            ("-a + a * 2", []),
            ("a ** 0.5", []),
            ("a % 3.0", []),
            ("exp(a)", []),
            # Only the inner c - d is zero; the outer divisor is then not evaluated.
            ("1.0 / (c - d)", ["c - d"]),
            ("a / (-b) + a", []),
        ],
    )
    def test_a_divisor_is_blocking_only_when_its_word_is_zero(
        self, divisor: str, zero_divisors: list[str]
    ) -> None:
        neuron = EquationNeuron(
            equations={"x": f"x / ({divisor})"},
            parameters={"a": 20.0, "b": 10.0, "c": 0.001, "d": 0.0015},
            method="map",
            dt=1.0,
        )
        contract = hardware_numeric_contract(neuron, Q88)
        divisors = [q for q in contract.quantities if q.kind == "divisor"]
        assert [q.name for q in divisors] == [f"equation x: {name}" for name in zero_divisors]
        assert all(
            (q.rtl_value, q.status, q.blocking) == (0.0, "underflows_to_zero", True)
            for q in divisors
        )

    def test_the_product_rounding_is_the_compiled_one(self) -> None:
        """(1/20 - 1/10)**2 = 0.0025: -13 * -13 = 169 LSB**2 truncates to 0, rounds to 1."""
        neuron = EquationNeuron(
            equations={"x": "x / ((1.0 / a - 1.0 / b) * (1.0 / a - 1.0 / b))"},
            parameters={"a": 20.0, "b": 10.0},
            method="map",
            dt=1.0,
        )
        (divisor,) = [
            q for q in hardware_numeric_contract(neuron, Q88).quantities if q.kind == "divisor"
        ]
        assert divisor.value == pytest.approx(0.0025)
        nearest = hardware_numeric_contract(neuron, Q88, rounding="nearest")
        assert not [q for q in nearest.quantities if q.kind == "divisor"]
        assert hardware_numeric_contract(neuron, Q1616).representable is True
