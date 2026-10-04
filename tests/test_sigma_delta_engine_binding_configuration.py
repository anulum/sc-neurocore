# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Configured native sampled APSDM state contracts

"""Compare configured native APSDM traces and atomic refusal with Python."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.models.sigma_delta import SigmaDeltaNeuron
from tests.engine_requirement import require_engine

extension = require_engine()
import sc_neurocore_engine

FIELDS = ("sigma", "reconstruction", "delta", "tau_reconstruction", "dt")
DEFAULTS = (0.0, 0.0, 1.0, 10.0, 0.1)
PROFILES = [
    {},
    {"sigma": 0.49},
    {"reconstruction": 0.4},
    {"delta": 0.5},
    {"tau_reconstruction": 5.0},
    {"dt": 0.2},
    {"sigma": -1.0, "reconstruction": -0.2},
    {"sigma": 0.5},
    {"sigma": -1e12},
    {"reconstruction": -1e12},
    {"sigma": 0.2, "reconstruction": -0.1, "delta": 0.8, "tau_reconstruction": 7.0, "dt": 0.07},
    {"dt": 1e-9},
]


class NativeCell(Protocol):
    """Describe the installed APSDM state and transition interface."""

    def step(self, current: float) -> int:
        """Advance a finite current sample or refuse without committing state."""
        ...

    def reset(self) -> None:
        """Clear the two dynamic states while retaining configuration."""
        ...

    def get_state(self) -> dict[str, float]:
        """Return a detached dictionary of both dynamic states."""
        ...


_CONSTRUCTOR = cast(Callable[..., NativeCell], sc_neurocore_engine.SigmaDeltaNeuron)
_BATCH = cast(Callable[..., dict[str, object]], extension.py_sigma_delta_simulate)


def _parameters(overrides: dict[str, float]) -> dict[str, float]:
    """Return all five constructor fields in their public positional order."""
    return dict(zip(FIELDS, DEFAULTS, strict=True)) | overrides


def _drive() -> npt.NDArray[np.float64]:
    """Return a readonly alternating current that crosses APSDM thresholds."""
    drive = np.tile([0.0, 2.0, 4.0, 0.2], 64)
    drive.setflags(write=False)
    return drive


@pytest.mark.parametrize("overrides", PROFILES)
def test_configured_class_batch_and_reset_match_python(overrides: dict[str, float]) -> None:
    """Compare complete traces, positional construction and configured reset."""
    parameters = _parameters(overrides)
    reference = SigmaDeltaNeuron(**parameters)
    cell, positional = _CONSTRUCTOR(**parameters), _CONSTRUCTOR(*parameters.values())
    drive = _drive()
    expected = []
    for current in drive:
        event = reference.step(float(current))
        assert cell.step(float(current)) == positional.step(float(current)) == event
        expected.append((reference.sigma, reference.reconstruction, event))
        np.testing.assert_allclose(
            [cell.get_state()[key] for key in ("sigma", "reconstruction")],
            expected[-1][:2],
            rtol=0.0,
            atol=2e-12,
        )
    batch = _BATCH(*parameters.values(), drive)
    assert set(batch) == {
        "sigma",
        "reconstruction",
        "events",
        "sigma_final",
        "reconstruction_final",
    }
    for index, key in enumerate(("sigma", "reconstruction", "events")):
        output = np.asarray(batch[key])
        assert output.shape == drive.shape
        assert output.dtype == (np.int32 if key == "events" else np.float64)
        if key == "events":
            np.testing.assert_array_equal(output, [row[index] for row in expected])
        else:
            np.testing.assert_allclose(
                output, [row[index] for row in expected], rtol=0.0, atol=2e-12
            )
    for key, value in zip(("sigma_final", "reconstruction_final"), expected[-1][:2], strict=True):
        assert float(cast(float, batch[key])) == pytest.approx(value, rel=0.0, abs=2e-12)
    detached = cell.get_state()
    detached["sigma"] = 99.0
    assert cell.get_state() == positional.get_state()
    np.testing.assert_array_equal(drive, np.tile([0.0, 2.0, 4.0, 0.2], 64))
    assert not drive.flags.writeable
    reference.reset()
    assert cast(Callable[[], object], cell.reset)() is None
    assert cast(Callable[[], object], positional.reset)() is None
    assert cell.get_state() == {"sigma": 0.0, "reconstruction": 0.0}
    for current in drive[:32]:
        assert (
            cell.step(float(current))
            == positional.step(float(current))
            == reference.step(float(current))
        )
    np.testing.assert_allclose(
        [cell.get_state()[key] for key in ("sigma", "reconstruction")],
        [reference.sigma, reference.reconstruction],
        rtol=0.0,
        atol=2e-12,
    )


@pytest.mark.parametrize("overrides", PROFILES)
def test_empty_batch_preserves_complete_initial_state(overrides: dict[str, float]) -> None:
    """Validate custom configuration and retain both finals for zero samples."""
    parameters = _parameters(overrides)
    batch = _BATCH(**parameters, currents=np.array([], dtype=np.float64))
    for key in ("sigma", "reconstruction", "events"):
        assert np.asarray(batch[key]).shape == (0,)
        assert np.asarray(batch[key]).dtype == (np.int32 if key == "events" else np.float64)
    assert batch["sigma_final"] == parameters["sigma"]
    assert batch["reconstruction_final"] == parameters["reconstruction"]


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_configuration_refuses_all_batch_lengths(field: str, value: float) -> None:
    """Reject each nonfinite field before empty or nonempty simulation."""
    parameters = _parameters({field: value})
    with pytest.raises(ValueError) as error:
        _CONSTRUCTOR(**parameters)
    assert str(error.value) == "invalid SigmaDelta state or configuration"
    for drive in (np.array([], dtype=np.float64), np.array([2.0, 0.2])):
        before = drive.copy()
        with pytest.raises(ValueError) as error:
            _BATCH(*parameters.values(), drive)
        assert str(error.value) == "invalid SigmaDelta state or configuration"
        np.testing.assert_array_equal(drive, before)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sigma": -1e12 - 1},
        {"sigma": 1e12 + 1},
        {"reconstruction": -1e12 - 1},
        {"reconstruction": 1e12 + 1},
        {"delta": 0.0},
        {"delta": -1.0},
        {"tau_reconstruction": 0.0},
        {"tau_reconstruction": -1.0},
        {"dt": 0.0},
        {"dt": -1.0},
    ],
)
def test_invalid_configuration_precedes_input_layout(overrides: dict[str, float]) -> None:
    """Retain configuration refusal before a reversed validly typed array."""
    parameters = _parameters(overrides)
    with pytest.raises(ValueError) as error:
        _CONSTRUCTOR(**parameters)
    assert str(error.value) == "invalid SigmaDelta state or configuration"
    drive = np.array([2.0, 0.2])[::-1]
    before = drive.copy()
    with pytest.raises(ValueError) as error:
        _BATCH(*parameters.values(), drive)
    assert str(error.value) == "invalid SigmaDelta state or configuration"
    np.testing.assert_array_equal(drive, before)


@pytest.mark.parametrize("current", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_step_and_late_batch_refusal_are_atomic(current: float) -> None:
    """Preserve class state, readonly inputs and the next valid transition."""
    cell, control = _CONSTRUCTOR(), _CONSTRUCTOR()
    assert cell.step(2.0) == control.step(2.0)
    before = cell.get_state()
    message = "invalid SigmaDelta state, configuration, or current"
    with pytest.raises(ValueError) as error:
        cell.step(current)
    assert str(error.value) == message and cell.get_state() == before
    drive = np.array([2.0, current])
    drive.setflags(write=False)
    with pytest.raises(ValueError) as error:
        _BATCH(*DEFAULTS, drive)
    assert str(error.value) == message
    np.testing.assert_array_equal(drive, [2.0, current])
    assert not drive.flags.writeable
    assert cell.step(0.2) == control.step(0.2)
    assert cell.get_state() == control.get_state()


@pytest.mark.parametrize(
    ("overrides", "current"),
    [
        ({"sigma": 1e12}, 1.0),
        ({"dt": 1e308}, 2.0),
        ({"sigma": 1e12, "delta": 1.5e12}, 0.0),
    ],
)
def test_candidate_refusal_preserves_state_and_reset_recovers(
    overrides: dict[str, float],
    current: float,
) -> None:
    """Reject overflow, bounded-state exit or post-event reconstruction atomically."""
    parameters = _parameters(overrides)
    cell, control = _CONSTRUCTOR(**parameters), _CONSTRUCTOR(**parameters)
    before = cell.get_state()
    with pytest.raises(ValueError) as error:
        cell.step(current)
    assert str(error.value) == "SigmaDelta candidate outside safety envelope"
    assert cell.get_state() == before
    drive = np.array([current])
    with pytest.raises(ValueError) as error:
        _BATCH(*parameters.values(), drive)
    assert str(error.value) == "SigmaDelta candidate outside safety envelope"
    np.testing.assert_array_equal(drive, [current])
    cell.reset()
    control.reset()
    assert cell.step(0.0) == control.step(0.0)
    assert cell.get_state() == control.get_state()
