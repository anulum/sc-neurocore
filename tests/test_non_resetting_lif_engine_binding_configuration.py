# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Configured native MAT(1) state and runtime contracts

"""Compare complete configured class, batch and five-runtime MAT(1) traces."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.models.non_resetting_lif import NonResettingLIFNeuron
from tests.engine_requirement import require_engine

extension = require_engine()
import sc_neurocore_engine

FIELDS = (
    "v",
    "theta",
    "refractory_remaining",
    "omega",
    "tau_m",
    "tau_theta",
    "alpha",
    "resistance",
    "refractory_period",
    "dt",
)
DEFAULTS = (0.0, 0.0, 0.0, 19.0, 5.0, 50.0, 37.0, 50.0, 2.0, 0.001)
PROFILES = [
    {},
    {"v": 12.0},
    {"theta": 4.0},
    {"refractory_remaining": 0.0005},
    {"omega": 15.0},
    {"tau_m": 7.0},
    {"tau_theta": 25.0},
    {"alpha": 18.0},
    {"resistance": 35.0},
    {"refractory_period": 1.0},
    {"dt": 0.01},
    {"v": 20.0},
    {"v": -200.0},
    {"refractory_remaining": 2.0},
    {"theta": 1e9, "omega": 1e9, "alpha": 1e9},
    {
        "v": -10.0,
        "theta": 5.0,
        "refractory_remaining": 0.03,
        "omega": 12.0,
        "tau_m": 7.0,
        "tau_theta": 30.0,
        "alpha": 20.0,
        "resistance": 40.0,
        "refractory_period": 0.5,
        "dt": 0.01,
    },
]


class NativeCell(Protocol):
    """Describe the public native MAT(1) temporal state."""

    def step(self, current: float) -> int:
        """Advance one valid current sample or refuse atomically."""
        ...

    def reset(self) -> None:
        """Zero dynamic state while retaining the configured model."""
        ...

    def get_state(self) -> dict[str, float]:
        """Return a detached dictionary of all three state values."""
        ...


_CONSTRUCTOR = cast(Callable[..., NativeCell], sc_neurocore_engine.NonResettingLIFNeuron)
_BATCH = cast(Callable[..., dict[str, object]], extension.py_non_resetting_lif_simulate)


def _parameters(overrides: dict[str, float]) -> dict[str, float]:
    """Return the complete independently pinned native constructor contract."""
    return dict(zip(FIELDS, DEFAULTS, strict=True)) | overrides


def _drive() -> npt.NDArray[np.float64]:
    """Return a readonly alternating current exercising configured transitions."""
    drive = np.tile([0.0, 0.7, 1.1, 0.2], 64)
    drive.setflags(write=False)
    return drive


@pytest.mark.parametrize("overrides", PROFILES)
def test_configured_native_class_batch_and_reset_match_python(overrides: dict[str, float]) -> None:
    """Compare every state sample, positional construction and reset continuation."""
    parameters = _parameters(overrides)
    reference = NonResettingLIFNeuron(**parameters)
    cell = _CONSTRUCTOR(**parameters)
    positional = _CONSTRUCTOR(*parameters.values())
    drive = _drive()
    expected = []
    for current in drive:
        event = reference.step(float(current))
        assert cell.step(float(current)) == positional.step(float(current)) == event
        expected.append((reference.v, reference.theta, reference.refractory_remaining, event))
        np.testing.assert_allclose(
            [cell.get_state()[key] for key in ("v", "theta", "refractory_remaining")],
            expected[-1][:3],
            rtol=0.0,
            atol=2e-12,
        )
    batch = _BATCH(*parameters.values(), drive)
    assert set(batch) == {
        "voltages",
        "theta",
        "refractory",
        "events",
        "v_final",
        "theta_final",
        "refractory_final",
    }
    for index, key in enumerate(("voltages", "theta", "refractory", "events")):
        output = np.asarray(batch[key])
        assert output.shape == drive.shape
        assert output.dtype == (np.int32 if key == "events" else np.float64)
        if key == "events":
            np.testing.assert_array_equal(output, [row[index] for row in expected])
        else:
            np.testing.assert_allclose(
                output, [row[index] for row in expected], rtol=0.0, atol=2e-12
            )
    for key, expected_value in zip(
        ("v_final", "theta_final", "refractory_final"),
        expected[-1][:3],
        strict=True,
    ):
        assert float(cast(float, batch[key])) == pytest.approx(expected_value, rel=0.0, abs=2e-12)
    detached = cell.get_state()
    detached["v"] = 99.0
    assert cell.get_state() == positional.get_state()
    reference.reset()
    cell.reset()
    positional.reset()
    assert cell.get_state() == {"v": 0.0, "theta": 0.0, "refractory_remaining": 0.0}
    for current in drive[:32]:
        assert (
            cell.step(float(current))
            == positional.step(float(current))
            == reference.step(float(current))
        )
    np.testing.assert_allclose(
        [cell.get_state()[key] for key in ("v", "theta", "refractory_remaining")],
        [reference.v, reference.theta, reference.refractory_remaining],
        rtol=0.0,
        atol=2e-12,
    )


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_configuration_refuses_class_and_all_batch_lengths(
    field: str, value: float
) -> None:
    """Refuse each nonfinite field before either empty or nonempty simulation."""
    parameters = _parameters({field: value})
    with pytest.raises(ValueError) as error:
        _CONSTRUCTOR(**parameters)
    assert str(error.value) == "invalid MAT(1) state or configuration"
    for drive in (np.array([], dtype=np.float64), np.array([0.7, 0.2])):
        before = drive.copy()
        with pytest.raises(ValueError) as error:
            _BATCH(*parameters.values(), drive)
        assert str(error.value) == "invalid MAT(1) state or configuration"
        np.testing.assert_array_equal(drive, before)


@pytest.mark.parametrize(
    "overrides",
    [
        {"v": -201.0},
        {"v": 201.0},
        {"theta": -1.0},
        {"theta": 1e9 + 1},
        {"omega": -1e9 - 1},
        {"omega": 1e9 + 1},
        {"alpha": -1.0},
        {"alpha": 1e9 + 1},
        {"tau_m": 0.0},
        {"tau_m": -1.0},
        {"tau_theta": 0.0},
        {"tau_theta": -1.0},
        {"resistance": 0.0},
        {"dt": 0.0},
        {"refractory_period": -1.0},
        {"refractory_remaining": -1.0},
        {"refractory_remaining": 2.01},
    ],
)
def test_invalid_configuration_precedes_array_layout_validation(
    overrides: dict[str, float],
) -> None:
    """Retain configuration refusal even when a validly typed input is reversed."""
    parameters = _parameters(overrides)
    with pytest.raises(ValueError) as error:
        _CONSTRUCTOR(**parameters)
    assert str(error.value) == "invalid MAT(1) state or configuration"
    drive = np.array([0.7, 0.2])[::-1]
    before = drive.copy()
    with pytest.raises(ValueError) as error:
        _BATCH(*parameters.values(), drive)
    assert str(error.value) == "invalid MAT(1) state or configuration"
    np.testing.assert_array_equal(drive, before)


@pytest.mark.parametrize("current", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_step_and_late_batch_refusal_are_atomic(current: float) -> None:
    """Preserve class state and readonly input after a refused late batch sample."""
    cell, control = _CONSTRUCTOR(), _CONSTRUCTOR()
    assert cell.step(0.7) == control.step(0.7)
    before = cell.get_state()
    message = "invalid NonResettingLIF state, configuration, or current"
    with pytest.raises(ValueError) as error:
        cell.step(current)
    assert str(error.value) == message and cell.get_state() == before
    drive = np.array([0.7, current])
    drive.setflags(write=False)
    with pytest.raises(ValueError) as error:
        _BATCH(*DEFAULTS, drive)
    assert str(error.value) == message
    np.testing.assert_array_equal(drive, [0.7, current])
    assert not drive.flags.writeable
    assert cell.step(0.2) == control.step(0.2)
    assert cell.get_state() == control.get_state()


@pytest.mark.parametrize(
    ("overrides", "current", "message"),
    [
        ({"v": 200.0, "dt": 1.0}, 1e308, "NonResettingLIF candidate outside safety envelope"),
        (
            {"v": 20.0, "theta": 1e9, "omega": -1e9, "alpha": 1e9},
            0.0,
            "NonResettingLIF post-spike threshold outside safety envelope",
        ),
    ],
)
def test_candidate_refusal_keeps_all_state_and_allows_reset_recovery(
    overrides: dict[str, float],
    current: float,
    message: str,
) -> None:
    """Refuse unsafe voltage or post-spike history without committing partial state."""
    parameters = _parameters(overrides)
    cell, control = _CONSTRUCTOR(**parameters), _CONSTRUCTOR(**parameters)
    before = cell.get_state()
    with pytest.raises(ValueError) as error:
        cell.step(current)
    assert str(error.value) == message and cell.get_state() == before
    drive = np.array([current])
    with pytest.raises(ValueError) as error:
        _BATCH(*parameters.values(), drive)
    assert str(error.value) == message
    np.testing.assert_array_equal(drive, [current])
    cell.reset()
    control.reset()
    assert cell.step(0.0) == control.step(0.0)
    assert cell.get_state() == control.get_state()
