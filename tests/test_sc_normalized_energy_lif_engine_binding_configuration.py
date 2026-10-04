# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Configured retained normalized energy-LIF contracts

"""Exercise every configured native state and its exact-flow reference."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.models.sc_normalized_energy_lif import SCNormalizedEnergyLIFNeuron
from tests.engine_requirement import require_engine

extension = require_engine()
import sc_neurocore_engine

FIELDS = (
    "v",
    "epsilon",
    "v_rest",
    "v_reset",
    "v_threshold",
    "tau_m",
    "tau_e",
    "alpha",
    "epsilon_0",
    "resistance",
    "dt",
)
DEFAULTS = (-70.0, 1.0, -70.0, -70.0, -50.0, 10.0, 500.0, 0.1, 1.0, 1.0, 1.0)
PROFILES = [
    {},
    {"v": -65.0},
    {"epsilon": 0.5},
    {"v_rest": -68.0},
    {"v_reset": -72.0},
    {"v_threshold": -45.0},
    {"tau_m": 7.0},
    {"tau_e": 250.0},
    {"alpha": 0.0},
    {"epsilon": 0.5, "epsilon_0": 0.5},
    {"resistance": 1.2},
    {"dt": 0.1},
    {"epsilon": 0.0, "epsilon_0": 0.0},
    {"epsilon": 0.1, "epsilon_0": 0.1},
    {"alpha": 2.0},
    {"epsilon": 0.25, "tau_m": 10.0, "tau_e": 10.0},
    {"epsilon": 0.25, "tau_m": 10.0, "tau_e": 10.000001},
    {"epsilon": 0.25, "tau_m": 10.0, "tau_e": 9.999999},
    {"epsilon": 0.25, "tau_m": 10.0, "tau_e": 10.00000000001},
    {"dt": 10.0, "tau_m": 10.0, "tau_e": 10.0},
    {"v": -200.0},
    {"v": 100.0},
    {"v": -200.0, "v_rest": -200.0, "v_reset": -200.0, "v_threshold": -180.0},
    {"v_threshold": 101.0},
    {
        "v": -66.0,
        "epsilon": 0.4,
        "v_rest": -68.0,
        "v_reset": -72.0,
        "v_threshold": -48.0,
        "tau_m": 8.0,
        "tau_e": 200.0,
        "alpha": 0.15,
        "epsilon_0": 0.8,
        "resistance": 1.1,
        "dt": 0.5,
    },
]
INVALID_CONFIGURATIONS = [
    {"v": -201.0},
    {"v": 101.0},
    {"epsilon": -0.1},
    {"epsilon": 1.1},
    {"v_reset": -201.0},
    {"v_reset": 101.0},
    {"epsilon_0": -1.0},
    {"alpha": -1.0},
    {"tau_m": 0.0},
    {"tau_m": -1.0},
    {"tau_e": 0.0},
    {"tau_e": -1.0},
    {"resistance": 0.0},
    {"resistance": -1.0},
    {"dt": 0.0},
    {"dt": -1.0},
    {"dt": 11.0},
    {"dt": 501.0},
    {"v_threshold": -70.0},
    {"v_reset": -40.0},
    {"epsilon_0": 0.5},
]
INVALID_REST_CONFIGURATIONS = [{"v_rest": -201.0}, {"v_rest": 101.0, "v_threshold": 102.0}]


class NativeCell(Protocol):
    """Describe the public normalized-energy native state."""

    def step(self, current: float) -> int:
        """Advance a current or refuse without either state mutation."""
        ...

    def reset(self) -> None:
        """Validate and restore the configured resting state."""
        ...

    def get_state(self) -> dict[str, float]:
        """Return an independent mapping of both dynamic values."""
        ...


_CONSTRUCTOR = cast(Callable[..., NativeCell], sc_neurocore_engine.SCNormalizedEnergyLIFNeuron)
_BATCH = cast(Callable[..., dict[str, object]], extension.py_sc_normalized_energy_lif_simulate)


def _parameters(overrides: dict[str, float]) -> dict[str, float]:
    """Return the independently pinned complete constructor field order."""
    return dict(zip(FIELDS, DEFAULTS, strict=True)) | overrides


def _drive() -> npt.NDArray[np.float64]:
    """Return the readonly frozen drive, exercising reset and depleted energy."""
    values = np.tile([30.0, 0.0, 50.0, 10.0], 64)
    values.setflags(write=False)
    return values


@pytest.mark.parametrize("overrides", PROFILES)
def test_configured_class_batch_and_reset_match_python(overrides: dict[str, float]) -> None:
    """Compare positional/keyword instances, every batch sample and reset continuation."""
    parameters = _parameters(overrides)
    reference = SCNormalizedEnergyLIFNeuron(**parameters)
    native, positional = _CONSTRUCTOR(**parameters), _CONSTRUCTOR(*parameters.values())
    drive = _drive()
    trace = []
    for current in drive:
        event = reference.step(float(current))
        assert native.step(float(current)) == positional.step(float(current)) == event
        trace.append((reference.v, reference.epsilon, event))
        np.testing.assert_allclose(
            [native.get_state()[k] for k in ("v", "epsilon")], trace[-1][:2], rtol=0, atol=2e-12
        )
    batch = _BATCH(*parameters.values(), drive)
    assert set(batch) == {"voltages", "epsilon", "events", "v_final", "epsilon_final"}
    expected = np.asarray(trace)
    for index, key in enumerate(("voltages", "epsilon", "events")):
        output = np.asarray(batch[key])
        assert output.shape == drive.shape and output.dtype == (
            np.int32 if key == "events" else np.float64
        )
        if key == "events":
            np.testing.assert_array_equal(output, expected[:, index])
        else:
            np.testing.assert_allclose(output, expected[:, index], rtol=0, atol=2e-12)
    for key, value in zip(("v_final", "epsilon_final"), trace[-1][:2], strict=True):
        assert float(cast(float, batch[key])) == pytest.approx(value, rel=0, abs=2e-12)
    np.testing.assert_array_equal(drive, np.tile([30.0, 0.0, 50.0, 10.0], 64))
    assert not drive.flags.writeable
    detached = native.get_state()
    detached["v"] = 99.0
    assert native.get_state() == positional.get_state()
    reference.reset()
    assert cast(Callable[[], object], native.reset)() is None
    assert cast(Callable[[], object], positional.reset)() is None
    assert native.get_state() == {"v": parameters["v_rest"], "epsilon": parameters["epsilon_0"]}
    for current in drive[:32]:
        assert (
            native.step(float(current))
            == positional.step(float(current))
            == reference.step(float(current))
        )
    np.testing.assert_allclose(
        [native.get_state()[k] for k in ("v", "epsilon")],
        [reference.v, reference.epsilon],
        rtol=0,
        atol=2e-12,
    )


@pytest.mark.parametrize("overrides", PROFILES)
def test_empty_batch_retains_exact_initial_state(overrides: dict[str, float]) -> None:
    """Empty owning arrays and exact initial finals retain every configuration."""
    parameters = _parameters(overrides)
    result = _BATCH(*parameters.values(), np.array([], dtype=np.float64))
    for key in ("voltages", "epsilon", "events"):
        assert np.asarray(result[key]).shape == (0,)
    assert result["v_final"] == parameters["v"] and result["epsilon_final"] == parameters["epsilon"]


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_configuration_refuses_class_and_all_batch_lengths(
    field: str, value: float
) -> None:
    """Every field refuses nonfinite values before even an empty batch."""
    parameters = _parameters({field: value})
    with pytest.raises(ValueError) as error:
        _CONSTRUCTOR(**parameters)
    assert str(error.value) == "invalid SC normalized EnergyLIF"
    for drive in (np.array([], dtype=np.float64), np.array([30.0, 0.0])):
        before = drive.copy()
        with pytest.raises(ValueError) as error:
            _BATCH(*parameters.values(), drive)
        assert str(error.value) == "invalid SC normalized EnergyLIF"
        np.testing.assert_array_equal(drive, before)


@pytest.mark.parametrize("overrides", INVALID_CONFIGURATIONS)
def test_invalid_configuration_precedes_array_layout(overrides: dict[str, float]) -> None:
    """Existing configuration refusal retains precedence over reversed input."""
    parameters = _parameters(overrides)
    with pytest.raises(ValueError) as error:
        _CONSTRUCTOR(**parameters)
    assert str(error.value) == "invalid SC normalized EnergyLIF"
    drive = np.array([30.0, 0.0])[::-1]
    with pytest.raises(ValueError) as error:
        _BATCH(*parameters.values(), drive)
    assert str(error.value) == "invalid SC normalized EnergyLIF"
    np.testing.assert_array_equal(drive, [0.0, 30.0])


@pytest.mark.parametrize("overrides", INVALID_REST_CONFIGURATIONS)
def test_invalid_rest_configuration_refuses_constructor_and_batch(
    overrides: dict[str, float],
) -> None:
    """Refuse formerly accepted resting voltages that cannot produce valid resets."""
    parameters = _parameters(overrides)
    with pytest.raises(ValueError) as error:
        _CONSTRUCTOR(**parameters)
    assert str(error.value) == "invalid SC normalized EnergyLIF"
    for drive in (np.array([], dtype=np.float64), np.array([30.0])):
        with pytest.raises(ValueError) as error:
            _BATCH(*parameters.values(), drive)
        assert str(error.value) == "invalid SC normalized EnergyLIF"


@pytest.mark.parametrize("current", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_step_and_late_batch_refusal_are_atomic(current: float) -> None:
    """Preserve both states and readonly late-failure input, then recover."""
    cell, control = _CONSTRUCTOR(), _CONSTRUCTOR()
    cell.step(30.0)
    control.step(30.0)
    before = cell.get_state()
    message = "invalid SC normalized EnergyLIF state, configuration, or current"
    with pytest.raises(ValueError) as error:
        cell.step(current)
    assert str(error.value) == message and cell.get_state() == before
    drive = np.array([30.0, current])
    drive.setflags(write=False)
    with pytest.raises(ValueError) as error:
        _BATCH(*DEFAULTS, drive)
    assert str(error.value) == message
    np.testing.assert_array_equal(drive, [30.0, current])
    assert not drive.flags.writeable
    assert cell.step(10.0) == control.step(10.0) and cell.get_state() == control.get_state()


@pytest.mark.parametrize("current", [1e308, 1e6])
def test_unsafe_candidate_is_atomic_and_reset_recovers(current: float) -> None:
    """Unsafe voltage candidates preserve both states and allow a reset retry."""
    cell, control = _CONSTRUCTOR(), _CONSTRUCTOR()
    before = cell.get_state()
    with pytest.raises(ValueError) as error:
        cell.step(current)
    assert str(error.value) == "SC normalized EnergyLIF candidate outside safety envelope"
    assert cell.get_state() == before
    drive = np.array([30.0, current])
    with pytest.raises(ValueError) as error:
        _BATCH(*DEFAULTS, drive)
    assert str(error.value) == "SC normalized EnergyLIF candidate outside safety envelope"
    np.testing.assert_array_equal(drive, [30.0, current])
    cell.reset()
    assert cell.step(30.0) == control.step(30.0) and cell.get_state() == control.get_state()


@pytest.mark.parametrize("rest", [-200.0, 100.0])
def test_inclusive_rest_bounds_reset_and_zero_current_remain_valid(rest: float) -> None:
    """Both resting endpoints support a valid reset and zero-current continuation."""
    cell = _CONSTRUCTOR(v_rest=rest, v_threshold=101.0)
    assert cast(Callable[[], object], cell.reset)() is None
    assert cell.get_state() == {"v": rest, "epsilon": 1.0}
    assert cell.step(0.0) == 0 and cell.get_state() == {"v": rest, "epsilon": 1.0}
