# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Configured native bipolar accumulator state contracts

"""Compare the complete retained accumulator contract through native APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.models.sc_sigma_delta_accumulator import SCSigmaDeltaAccumulatorNeuron
from tests.engine_requirement import require_engine

extension = require_engine()
import sc_neurocore_engine

FIELDS = ("sigma", "v_threshold")
DEFAULTS = (0.0, 1.0)
MAX_FINITE = float(np.finfo(np.float64).max)
PROFILES = [
    {},
    {"sigma": 0.75},
    {"v_threshold": 0.5},
    {"sigma": 3.25},
    {"sigma": -3.25},
    {"sigma": 1.0},
    {"sigma": -1.0},
    {"sigma": 0.2, "v_threshold": 0.7},
    {"sigma": MAX_FINITE},
    {"sigma": -MAX_FINITE},
    {"v_threshold": MAX_FINITE},
    {"v_threshold": float(np.nextafter(0.0, 1.0))},
    {"sigma": -0.0},
]


class NativeCell(Protocol):
    """Describe the installed accumulator's public state and transition API."""

    def step(self, current: float) -> int:
        """Advance one sample and return one signed event or refuse atomically."""
        ...

    def reset(self) -> None:
        """Clear the residual state while retaining the threshold."""
        ...

    def get_state(self) -> dict[str, float]:
        """Return a detached dictionary containing the complete residual state."""
        ...


_CONSTRUCTOR = cast(Callable[..., NativeCell], sc_neurocore_engine.SCSigmaDeltaAccumulatorNeuron)
_BATCH = cast(Callable[..., dict[str, object]], extension.py_sc_sigma_delta_accumulator_simulate)


def _parameters(overrides: dict[str, float]) -> dict[str, float]:
    """Return both constructor fields in their documented positional order."""
    return dict(zip(FIELDS, DEFAULTS, strict=True)) | overrides


def _drive() -> npt.NDArray[np.float64]:
    """Return a readonly signed drive exercising carry and both event directions."""
    drive = np.tile([0.0, 3.25, -4.5, 0.2], 64)
    drive.setflags(write=False)
    return drive


@pytest.mark.parametrize("overrides", PROFILES)
def test_configured_class_batch_and_reset_match_python(overrides: dict[str, float]) -> None:
    """Compare all samples, custom initial residuals and configured reset traces."""
    parameters = _parameters(overrides)
    reference = SCSigmaDeltaAccumulatorNeuron(**parameters)
    cell, positional = _CONSTRUCTOR(**parameters), _CONSTRUCTOR(*parameters.values())
    assert cell.get_state() == {"sigma": parameters["sigma"]}
    drive = _drive()
    expected = []
    for current in drive:
        event = reference.step(float(current))
        assert cell.step(float(current)) == positional.step(float(current)) == event
        assert cell.get_state() == positional.get_state() == {"sigma": reference.sigma}
        expected.append((reference.sigma, event))
    batch = _BATCH(*parameters.values(), drive)
    assert set(batch) == {"sigma", "events", "sigma_final"}
    for index, key in enumerate(("sigma", "events")):
        output = np.asarray(batch[key])
        assert output.shape == drive.shape
        assert output.dtype == (np.int32 if key == "events" else np.float64)
        np.testing.assert_array_equal(output, [row[index] for row in expected])
    assert batch["sigma_final"] == reference.sigma
    detached = cell.get_state()
    detached["sigma"] = 99.0
    assert cell.get_state() == positional.get_state()
    np.testing.assert_array_equal(drive, np.tile([0.0, 3.25, -4.5, 0.2], 64))
    assert not drive.flags.writeable
    reference.reset()
    assert cast(Callable[[], object], cell.reset)() is None
    assert cast(Callable[[], object], positional.reset)() is None
    assert cell.get_state() == {"sigma": 0.0}
    for current in drive[:32]:
        assert (
            cell.step(float(current))
            == positional.step(float(current))
            == reference.step(float(current))
        )
        assert cell.get_state() == positional.get_state() == {"sigma": reference.sigma}


@pytest.mark.parametrize("overrides", PROFILES)
def test_empty_batch_retains_initial_state_and_signed_zero(overrides: dict[str, float]) -> None:
    """Retain the supplied finite residual, including its zero sign, for no samples."""
    parameters = _parameters(overrides)
    batch = _BATCH(**parameters, currents=np.array([], dtype=np.float64))
    for key in ("sigma", "events"):
        assert np.asarray(batch[key]).shape == (0,)
        assert np.asarray(batch[key]).dtype == (np.int32 if key == "events" else np.float64)
    final = float(cast(float, batch["sigma_final"]))
    assert final == parameters["sigma"]
    assert np.signbit(final) == np.signbit(parameters["sigma"])


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_configuration_refuses_all_batch_lengths(field: str, value: float) -> None:
    """Reject each nonfinite field before empty or nonempty batch execution."""
    parameters = _parameters({field: value})
    with pytest.raises(ValueError) as error:
        _CONSTRUCTOR(**parameters)
    assert str(error.value) == "invalid SC SigmaDelta accumulator"
    for drive in (np.array([], dtype=np.float64), np.array([3.25, -4.5])):
        before = drive.copy()
        with pytest.raises(ValueError) as error:
            _BATCH(*parameters.values(), drive)
        assert str(error.value) == "invalid SC SigmaDelta accumulator"
        np.testing.assert_array_equal(drive, before)


@pytest.mark.parametrize("threshold", [0.0, -0.0, -1.0])
def test_nonpositive_threshold_precedes_array_layout(threshold: float) -> None:
    """Preserve threshold refusal before checking a reversed float64 input."""
    parameters = _parameters({"v_threshold": threshold})
    with pytest.raises(ValueError) as error:
        _CONSTRUCTOR(**parameters)
    assert str(error.value) == "invalid SC SigmaDelta accumulator"
    drive = np.array([3.25, -4.5])[::-1]
    before = drive.copy()
    with pytest.raises(ValueError) as error:
        _BATCH(*parameters.values(), drive)
    assert str(error.value) == "invalid SC SigmaDelta accumulator"
    np.testing.assert_array_equal(drive, before)


@pytest.mark.parametrize("current", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_step_and_late_batch_refusal_are_atomic(current: float) -> None:
    """Preserve signed residuals, readonly inputs and the next valid transition."""
    cell, control = _CONSTRUCTOR(), _CONSTRUCTOR()
    assert cell.step(3.25) == control.step(3.25)
    before = cell.get_state()
    message = "invalid SC SigmaDelta accumulator state or current"
    with pytest.raises(ValueError) as error:
        cell.step(current)
    assert str(error.value) == message and cell.get_state() == before
    drive = np.array([3.25, current])
    drive.setflags(write=False)
    with pytest.raises(ValueError) as error:
        _BATCH(*DEFAULTS, drive)
    assert str(error.value) == message
    np.testing.assert_array_equal(drive, [3.25, current])
    assert not drive.flags.writeable
    assert cell.step(-4.5) == control.step(-4.5)
    assert cell.get_state() == control.get_state()


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_candidate_overflow_preserves_state_and_reset_recovers(sign: float) -> None:
    """Refuse either signed overflow without discarding the finite initial residual."""
    parameters = _parameters({"sigma": sign * MAX_FINITE})
    cell, control = _CONSTRUCTOR(**parameters), _CONSTRUCTOR(**parameters)
    current = sign * MAX_FINITE
    before = cell.get_state()
    with pytest.raises(ValueError) as error:
        cell.step(current)
    assert str(error.value) == "SC SigmaDelta accumulator candidate is non-finite"
    assert cell.get_state() == before
    drive = np.array([current])
    with pytest.raises(ValueError) as error:
        _BATCH(*parameters.values(), drive)
    assert str(error.value) == "SC SigmaDelta accumulator candidate is non-finite"
    np.testing.assert_array_equal(drive, [current])
    cell.reset()
    control.reset()
    assert cell.step(0.3) == control.step(0.3)
    assert cell.get_state() == control.get_state()
