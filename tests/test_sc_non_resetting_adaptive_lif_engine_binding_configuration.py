# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Configured retained adaptive LIF native contracts

"""Compare complete native configurations with the retained Python recurrence."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.models.sc_non_resetting_adaptive_lif import (
    SCNonResettingAdaptiveLIFNeuron,
)
from tests.engine_requirement import require_engine

extension = require_engine()
import sc_neurocore_engine

FIELDS = ("v", "theta", "v_rest", "theta_rest", "delta_theta", "tau_m", "tau_theta", "r_m", "dt")
DEFAULTS = (-65.0, -50.0, -65.0, -50.0, 5.0, 10.0, 50.0, 1.0, 0.1)
PROFILES = [
    {},
    {"v": -64.0},
    {"theta": -49.0},
    {"v_rest": -63.0},
    {"theta_rest": -48.0},
    {"delta_theta": 0.0},
    {"delta_theta": 8.0},
    {"tau_m": 3.0},
    {"tau_theta": 15.0},
    {"r_m": 0.0},
    {"r_m": 1.5},
    {"dt": 0.5},
    {"v": -201.0},
    {"theta": 201.0},
    {"v_rest": -500.0, "theta_rest": 500.0},
    {"dt": 10.0, "tau_m": 10.0, "tau_theta": 10.0},
    {"v": 1e20, "v_rest": 0.0, "delta_theta": 0.0, "dt": 40.0, "tau_m": 1.0, "tau_theta": 1.0},
    {"dt": 1000.0, "tau_m": 1.0, "tau_theta": 1.0},
    {"dt": 1e308, "tau_m": 1e-308, "tau_theta": 1e-308},
    {"dt": 5e-324, "tau_m": 1e308, "tau_theta": 1e308},
    {"dt": 1e-308, "tau_m": 1e-308, "tau_theta": 1e-308},
    {
        "v": -50.0,
        "theta": -50.0,
        "v_rest": -50.0,
        "theta_rest": -50.0,
        "r_m": 0.0,
        "delta_theta": 0.0,
    },
    {
        "v": -64.0,
        "theta": -49.0,
        "v_rest": -62.0,
        "theta_rest": -47.0,
        "delta_theta": 3.0,
        "tau_m": 8.0,
        "tau_theta": 30.0,
        "r_m": 1.2,
        "dt": 0.3,
    },
]
INVALID_CONFIGURATIONS = [
    {"delta_theta": -1.0},
    {"r_m": -1.0},
    {"tau_m": 0.0},
    {"tau_m": -1.0},
    {"tau_theta": 0.0},
    {"tau_theta": -1.0},
    {"dt": 0.0},
    {"dt": -1.0},
]


class NativeCell(Protocol):
    """Describe the public native transition and detached state interface."""

    def step(self, current: float) -> int:
        """Advance one sample or refuse without state mutation."""
        ...

    def reset(self) -> None:
        """Restore the configured rests without altering configuration."""
        ...

    def get_state(self) -> dict[str, float]:
        """Return a detached dictionary of voltage and threshold."""
        ...


_CONSTRUCTOR = cast(Callable[..., NativeCell], sc_neurocore_engine.SCNonResettingAdaptiveLIFNeuron)
_BATCH = cast(Callable[..., dict[str, object]], extension.py_sc_non_resetting_adaptive_lif_simulate)


def _parameters(overrides: dict[str, float]) -> dict[str, float]:
    """Return all nine parameters in the independently pinned public order."""
    return dict(zip(FIELDS, DEFAULTS, strict=True)) | overrides


def _drive() -> npt.NDArray[np.float64]:
    """Return a readonly drive containing relaxation and threshold events."""
    drive = np.tile([20.0, 0.0, 60.0, 10.0], 64)
    drive.setflags(write=False)
    return drive


@pytest.mark.parametrize("overrides", PROFILES)
def test_configured_class_batch_and_reset_match_python(overrides: dict[str, float]) -> None:
    """Preserve positional and keyword construction, both traces and reset continuation."""
    parameters = _parameters(overrides)
    reference = SCNonResettingAdaptiveLIFNeuron(**parameters)
    native, positional = _CONSTRUCTOR(**parameters), _CONSTRUCTOR(*parameters.values())
    drive = _drive()
    trace = []
    for current in drive:
        event = reference.step(float(current))
        assert native.step(float(current)) == positional.step(float(current)) == event
        trace.append((reference.v, reference.theta, event))
        np.testing.assert_allclose(
            [native.get_state()[k] for k in ("v", "theta")], trace[-1][:2], rtol=0, atol=2e-12
        )
    batch = _BATCH(*parameters.values(), drive)
    assert set(batch) == {"voltages", "theta", "events", "v_final", "theta_final"}
    expected = np.asarray(trace)
    for index, key in enumerate(("voltages", "theta", "events")):
        output = np.asarray(batch[key])
        assert output.shape == drive.shape and output.dtype == (
            np.int32 if key == "events" else np.float64
        )
        if key == "events":
            np.testing.assert_array_equal(output, expected[:, index])
        else:
            np.testing.assert_allclose(output, expected[:, index], rtol=0, atol=2e-12)
    assert batch["v_final"] == reference.v and batch["theta_final"] == reference.theta
    detached = native.get_state()
    detached["v"] = 99.0
    assert native.get_state() == positional.get_state()
    reference.reset()
    assert cast(Callable[[], object], native.reset)() is None
    assert cast(Callable[[], object], positional.reset)() is None
    assert native.get_state() == {"v": parameters["v_rest"], "theta": parameters["theta_rest"]}
    for current in drive[:32]:
        assert (
            native.step(float(current))
            == positional.step(float(current))
            == reference.step(float(current))
        )
    np.testing.assert_allclose(
        [native.get_state()[k] for k in ("v", "theta")],
        [reference.v, reference.theta],
        rtol=0,
        atol=2e-12,
    )
    np.testing.assert_array_equal(drive, np.tile([20.0, 0.0, 60.0, 10.0], 64))
    assert not drive.flags.writeable


@pytest.mark.parametrize("overrides", PROFILES)
def test_empty_batch_retains_exact_initial_state(overrides: dict[str, float]) -> None:
    """Every accepted configuration returns empty typed arrays and exact initial finals."""
    parameters = _parameters(overrides)
    result = _BATCH(*parameters.values(), np.array([], dtype=np.float64))
    for key in ("voltages", "theta", "events"):
        assert np.asarray(result[key]).shape == (0,)
    assert result["v_final"] == parameters["v"] and result["theta_final"] == parameters["theta"]


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_configuration_refuses_class_and_all_batch_lengths(
    field: str, value: float
) -> None:
    """All state and parameter fields refuse before an empty or nonempty batch."""
    parameters = _parameters({field: value})
    with pytest.raises(ValueError, match="^invalid SC adaptive LIF state or configuration$"):
        _CONSTRUCTOR(**parameters)
    for drive in (np.array([], dtype=np.float64), np.array([20.0, 0.0])):
        before = drive.copy()
        with pytest.raises(ValueError, match="^invalid SC adaptive LIF state or configuration$"):
            _BATCH(*parameters.values(), drive)
        np.testing.assert_array_equal(drive, before)


@pytest.mark.parametrize("overrides", INVALID_CONFIGURATIONS)
def test_invalid_configuration_precedes_array_layout(overrides: dict[str, float]) -> None:
    """Finite domain errors retain their exact refusal and precedence over layout."""
    parameters = _parameters(overrides)
    with pytest.raises(ValueError, match="^invalid SC adaptive LIF state or configuration$"):
        _CONSTRUCTOR(**parameters)
    drive = np.array([20.0, 0.0])[::-1]
    with pytest.raises(ValueError, match="^invalid SC adaptive LIF state or configuration$"):
        _BATCH(*parameters.values(), drive)
    np.testing.assert_array_equal(drive, [0.0, 20.0])


@pytest.mark.parametrize("current", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_step_and_late_batch_refusal_are_atomic(current: float) -> None:
    """A late invalid current preserves readonly inputs and independent class state."""
    cell, control = _CONSTRUCTOR(), _CONSTRUCTOR()
    cell.step(20.0)
    control.step(20.0)
    before = cell.get_state()
    message = "invalid SC non-resetting adaptive LIF state, configuration, or current"
    with pytest.raises(ValueError) as error:
        cell.step(current)
    assert str(error.value) == message and cell.get_state() == before
    drive = np.array([20.0, current])
    drive.setflags(write=False)
    with pytest.raises(ValueError) as error:
        _BATCH(*DEFAULTS, drive)
    assert str(error.value) == message
    np.testing.assert_array_equal(drive, [20.0, current])
    assert not drive.flags.writeable
    assert cell.step(10.0) == control.step(10.0) and cell.get_state() == control.get_state()


@pytest.mark.parametrize(
    ("overrides", "current", "message"),
    [
        ({"r_m": 1e308}, 20.0, "SC non-resetting adaptive LIF steady state is non-finite"),
        (
            {"v": 1.5e308, "theta": 1e308, "theta_rest": 1e308, "delta_theta": 1e308},
            0.0,
            "SC non-resetting adaptive LIF threshold is non-finite",
        ),
    ],
)
def test_finite_overflow_is_atomic_and_reset_recovers(
    overrides: dict[str, float], current: float, message: str
) -> None:
    """Finite accepted parameters may refuse a candidate without committing either state."""
    parameters = _parameters(overrides)
    cell = _CONSTRUCTOR(**parameters)
    before = cell.get_state()
    with pytest.raises(ValueError) as error:
        cell.step(current)
    assert str(error.value) == message and cell.get_state() == before
    with pytest.raises(ValueError) as error:
        _BATCH(*parameters.values(), np.array([current]))
    assert str(error.value) == message
    cell.reset()
    assert cell.get_state() == {"v": parameters["v_rest"], "theta": parameters["theta_rest"]}
    assert cell.step(0.0) == 0
