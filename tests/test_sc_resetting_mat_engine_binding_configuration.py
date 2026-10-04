# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed SC resetting-MAT complete configuration and reset

"""Exercise installed configured neurons and complete source-reference traces."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import pytest

from tests.engine_requirement import require_engine

extension = require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.neurons.models.sc_resetting_mat import SCResettingMATNeuron

FIELDS = tuple(field.name for field in fields(SCResettingMATNeuron))
DEFAULTS = tuple(float(getattr(SCResettingMATNeuron(), field)) for field in FIELDS)
PROFILE = (-66.0, 2.0, 3.0, -70.0, -71.0, -49.0, 12.0, 15.0, 250.0, 4.0, 2.0, 1.25, 0.25)


def _batch(configuration: tuple[float, ...], currents: NDArray[Any]) -> dict[str, Any]:
    """Call the installed public batch with every configured field."""
    return cast(dict[str, Any], extension.py_sc_resetting_mat_simulate(*configuration, currents))


@pytest.mark.parametrize("field", range(len(FIELDS)))
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_complete_constructor_and_empty_batch_refusal(field: int, bad: float) -> None:
    """Invalid configuration cannot pass admission through an empty batch."""
    configuration = list(DEFAULTS)
    configuration[field] = bad
    with pytest.raises(ValueError, match="invalid SC resetting-MAT state or configuration"):
        engine.SCResettingMATNeuron(*configuration)
    drive = np.empty(0, dtype=np.float64)
    drive.setflags(write=False)
    with pytest.raises(ValueError, match="invalid SC resetting-MAT state or configuration"):
        _batch(tuple(configuration), drive)
    assert drive.size == 0 and not drive.flags.writeable


@pytest.mark.parametrize("rest", [-500.0, 500.0])
def test_admitted_configuration_reset_refuses_without_poisoning_live_class(rest: float) -> None:
    """A refused resting candidate preserves state and the next valid transition."""
    configuration = list(DEFAULTS)
    configuration[:3] = [-65.0, 2.0, 3.0]
    configuration[3] = rest
    cell = engine.SCResettingMATNeuron(*configuration)
    control = engine.SCResettingMATNeuron(*configuration)
    before = cell.get_state()
    with pytest.raises(ValueError, match="invalid SC resetting-MAT reset state or configuration"):
        cell.reset()
    assert cell.get_state() == before
    assert cell.step(20.0) == control.step(20.0)
    assert cell.get_state() == control.get_state()


@pytest.mark.parametrize("dt", [0.25, 1.0, 2.0])
@pytest.mark.parametrize("current", [0.0, 20.0, 50.0])
def test_configured_complete_batch_and_class_match_public_python(dt: float, current: float) -> None:
    """All configured state traces, events and final fields match the source."""
    configuration = (*PROFILE[:-1], dt)
    drive = np.resize(np.array([current, 0.0, 50.0, 10.0], dtype=np.float64), 256)
    before = drive.copy()
    drive.setflags(write=False)
    source = SCResettingMATNeuron(*configuration)
    cell = engine.SCResettingMATNeuron(*configuration)
    trace: list[tuple[float, float, float, int]] = []
    for sample in drive:
        event = source.step(float(sample))
        assert cell.step(float(sample)) == event
        state = cell.get_state()
        np.testing.assert_allclose(
            [state[k] for k in ("v", "theta1", "theta2")],
            [source.v, source.theta1, source.theta2],
            rtol=0,
            atol=2e-12,
        )
        trace.append((source.v, source.theta1, source.theta2, event))
    result = _batch(configuration, drive)
    for index, name in enumerate(("voltages", "theta1", "theta2", "events")):
        output = result[name]
        assert output.dtype == (np.int32 if name == "events" else np.float64)
        if name == "events":
            np.testing.assert_array_equal(output, [row[index] for row in trace])
        else:
            np.testing.assert_allclose(output, [row[index] for row in trace], rtol=0, atol=2e-12)
        assert output.flags.owndata and output.flags.writeable
        assert output.flags.c_contiguous and output.flags.aligned
        assert not np.shares_memory(output, drive)
    for name, value in (
        ("v_final", source.v),
        ("theta1_final", source.theta1),
        ("theta2_final", source.theta2),
    ):
        assert result[name] == pytest.approx(value, abs=2e-12)
    np.testing.assert_array_equal(drive, before)
    assert not drive.flags.writeable
    cell.reset()
    source.reset()
    assert cell.get_state() == {"v": source.v, "theta1": source.theta1, "theta2": source.theta2}
    assert cell.step(50.0) == source.step(50.0)
    for key in ("v", "theta1", "theta2"):
        assert cell.get_state()[key] == pytest.approx(getattr(source, key), abs=2e-12)


def test_empty_batch_retains_all_configured_dynamic_fields() -> None:
    """Empty traces are owning typed arrays with configured final values."""
    result = _batch(PROFILE, np.empty(0, dtype=np.float64))
    assert set(result) == {
        "voltages",
        "theta1",
        "theta2",
        "events",
        "v_final",
        "theta1_final",
        "theta2_final",
    }
    for name in ("voltages", "theta1", "theta2", "events"):
        output = result[name]
        assert output.shape == (0,)
        assert output.dtype == (np.int32 if name == "events" else np.float64)
        assert output.flags.owndata and output.flags.c_contiguous and output.flags.aligned
    assert (result["v_final"], result["theta1_final"], result["theta2_final"]) == PROFILE[:3]
