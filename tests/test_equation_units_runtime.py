# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the equation strict-unit runtime

"""Unit tests for the strict pint-backed dimensional-validation runtime.

Exercises :func:`prepare_strict_runtime` (dimensional validation plus the
conversion of pint quantities to base-unit floats) and
:func:`convert_runtime_value` (runtime input coercion) directly, covering both
the coherent-model happy path and every rejection branch.
"""

from __future__ import annotations

import pytest

from sc_neurocore.neurons._units import UNIT_REGISTRY, DimensionalError
from sc_neurocore.neurons.equation_units_runtime import (
    StrictRuntime,
    convert_runtime_value,
    prepare_strict_runtime,
)

MEGAOHM = 1e6 * UNIT_REGISTRY.ohm


def _coherent_lif_runtime() -> StrictRuntime:
    """Prepare the strict runtime for a dimensionally coherent LIF membrane."""
    return prepare_strict_runtime(
        equations={"v": "(-(v - E_L) + R * I) / tau_m"},
        threshold_expr="v > v_threshold",
        reset_rules={"v": "v_reset"},
        input_unit_name="I",
        raw_parameters={
            "E_L": -65.0 * UNIT_REGISTRY.millivolt,
            "R": 100.0 * MEGAOHM,
            "tau_m": 10.0 * UNIT_REGISTRY.millisecond,
        },
        raw_state={"v": -65.0 * UNIT_REGISTRY.millivolt},
        raw_constants={
            "v_threshold": -50.0 * UNIT_REGISTRY.millivolt,
            "v_reset": -65.0 * UNIT_REGISTRY.millivolt,
        },
        dt=0.1 * UNIT_REGISTRY.millisecond,
        input_unit=1.0 * UNIT_REGISTRY.nanoampere,
    )


def test_coherent_model_converts_to_base_unit_floats() -> None:
    """A coherent model yields base-unit floats and the full unit maps."""
    runtime = _coherent_lif_runtime()

    assert isinstance(runtime, StrictRuntime)
    # Base SI units: millivolt -> volt, so -65 mV -> -0.065 V.
    assert runtime.state["v"] == pytest.approx(-0.065)
    assert runtime.parameters["E_L"] == pytest.approx(-0.065)
    assert runtime.parameters["tau_m"] == pytest.approx(0.01)  # 10 ms -> 0.01 s
    assert runtime.constants["v_threshold"] == pytest.approx(-0.05)
    assert runtime.dt == pytest.approx(1e-4)  # 0.1 ms -> 1e-4 s
    # The input current and every named symbol carry a runtime unit.
    assert "I" in runtime.runtime_units
    assert set(runtime.runtime_units) >= {"I", "E_L", "R", "tau_m", "v", "v_threshold", "v_reset"}
    # State keeps a voltage-dimensioned base unit and the caller's display unit.
    assert runtime.base_state_units["v"].dimensionality == UNIT_REGISTRY.volt.dimensionality
    assert str(runtime.display_state_units["v"]) == "millivolt"


def test_model_without_input_skips_the_input_unit_requirement() -> None:
    """A model that never references ``I`` needs no ``input_unit``."""
    runtime = prepare_strict_runtime(
        equations={"v": "-v / tau_m"},
        threshold_expr="v > v_threshold",
        reset_rules={},
        input_unit_name="I",
        raw_parameters={"tau_m": 10.0 * UNIT_REGISTRY.millisecond},
        raw_state={"v": -65.0 * UNIT_REGISTRY.millivolt},
        raw_constants={"v_threshold": -50.0 * UNIT_REGISTRY.millivolt},
        dt=0.1 * UNIT_REGISTRY.millisecond,
        input_unit=None,
    )
    assert "I" not in runtime.runtime_units
    assert runtime.state["v"] == pytest.approx(-0.065)


def test_reset_rule_referencing_input_marks_input_used() -> None:
    """A reset rule that references ``I`` also requires a declared input unit."""
    runtime = prepare_strict_runtime(
        equations={"v": "-v / tau_m"},  # dynamics do not reference I
        threshold_expr=None,
        reset_rules={"v": "R * I"},  # I referenced only in the reset -> V
        input_unit_name="I",
        raw_parameters={
            "R": 100.0 * MEGAOHM,
            "tau_m": 10.0 * UNIT_REGISTRY.millisecond,
        },
        raw_state={"v": -65.0 * UNIT_REGISTRY.millivolt},
        raw_constants={},
        dt=0.1 * UNIT_REGISTRY.millisecond,
        input_unit=1.0 * UNIT_REGISTRY.nanoampere,
    )
    assert "I" in runtime.runtime_units


def test_missing_state_quantity_is_rejected() -> None:
    """Every equation variable needs an explicit state quantity in strict mode."""
    with pytest.raises(ValueError, match="requires explicit state quantities"):
        prepare_strict_runtime(
            equations={"v": "-v / tau_m", "w": "w"},
            threshold_expr=None,
            reset_rules={},
            input_unit_name="I",
            raw_parameters={"tau_m": 10.0 * UNIT_REGISTRY.millisecond},
            raw_state={"v": -65.0 * UNIT_REGISTRY.millivolt},
            raw_constants={},
            dt=0.1 * UNIT_REGISTRY.millisecond,
            input_unit=None,
        )


def test_input_referencing_model_requires_input_unit() -> None:
    """Referencing ``I`` without declaring ``input_unit`` is rejected."""
    with pytest.raises(ValueError, match="requires input_unit"):
        prepare_strict_runtime(
            equations={"v": "(-v + R * I) / tau_m"},
            threshold_expr=None,
            reset_rules={},
            input_unit_name="I",
            raw_parameters={
                "R": 100.0 * MEGAOHM,
                "tau_m": 10.0 * UNIT_REGISTRY.millisecond,
            },
            raw_state={"v": -65.0 * UNIT_REGISTRY.millivolt},
            raw_constants={},
            dt=0.1 * UNIT_REGISTRY.millisecond,
            input_unit=None,
        )


def test_incoherent_dimensions_raise_dimensional_error() -> None:
    """A dimensionally inconsistent right-hand side is refused."""
    with pytest.raises(DimensionalError):
        prepare_strict_runtime(
            equations={"v": "-v / tau_m"},
            threshold_expr=None,
            reset_rules={},
            input_unit_name="I",
            raw_parameters={"tau_m": 10.0 * UNIT_REGISTRY.picofarad},  # not a time
            raw_state={"v": -65.0 * UNIT_REGISTRY.millivolt},
            raw_constants={},
            dt=0.1 * UNIT_REGISTRY.millisecond,
            input_unit=None,
        )


def test_non_boolean_threshold_is_rejected() -> None:
    """A threshold expression that is not boolean-valued is refused."""
    with pytest.raises(ValueError, match="must evaluate to a boolean"):
        prepare_strict_runtime(
            equations={"v": "-v / tau_m"},
            threshold_expr="v",  # a quantity, not a comparison
            reset_rules={},
            input_unit_name="I",
            raw_parameters={"tau_m": 10.0 * UNIT_REGISTRY.millisecond},
            raw_state={"v": -65.0 * UNIT_REGISTRY.millivolt},
            raw_constants={},
            dt=0.1 * UNIT_REGISTRY.millisecond,
            input_unit=None,
        )


def test_convert_runtime_value_passthrough_when_not_strict() -> None:
    """Non-strict mode returns a plain float without touching units."""
    assert convert_runtime_value(strict_units=False, runtime_units={}, name="I", value=3.0) == 3.0


def test_convert_runtime_value_passthrough_for_plain_float_in_strict_mode() -> None:
    """A non-quantity value in strict mode is still returned as a float."""
    assert convert_runtime_value(strict_units=True, runtime_units={}, name="I", value=2.5) == 2.5


def test_convert_runtime_value_requires_declared_unit() -> None:
    """Converting a quantity for an undeclared name is refused."""
    with pytest.raises(ValueError, match="No runtime unit declared for 'I'"):
        convert_runtime_value(
            strict_units=True,
            runtime_units={},
            name="I",
            value=2.0 * UNIT_REGISTRY.nanoampere,
        )


def test_convert_runtime_value_coerces_quantity_to_base_float() -> None:
    """A quantity is converted to the base-unit magnitude of its declared unit."""
    runtime = _coherent_lif_runtime()
    converted = convert_runtime_value(
        strict_units=True,
        runtime_units=runtime.runtime_units,
        name="I",
        value=2.0 * UNIT_REGISTRY.nanoampere,
    )
    assert converted == pytest.approx(2e-9)  # 2 nA -> 2e-9 A
