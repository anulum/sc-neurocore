# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed source MAT complete configuration and reset

"""Exercise installed MAT classes and complete configured batch trajectories."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import pytest

from tests.engine_requirement import require_engine

extension = require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.neurons.models.mat import MATNeuron

FIELDS = tuple(field.name for field in fields(MATNeuron))
DEFAULTS = tuple(float(getattr(MATNeuron(), field)) for field in FIELDS)
PROFILE = (4.0, 2.0, 3.0, 0.5, 15.0, 8.0, 12.0, 250.0, 7.0, 3.0, 40.0, 1.0, 0.05)
DYNAMICS = ("v", "theta1", "theta2", "refractory_remaining")


def _batch(configuration: tuple[float, ...], currents: NDArray[Any]) -> dict[str, Any]:
    """Call the actual installed public batch with every configured field."""
    return cast(dict[str, Any], extension.py_mat_simulate(*configuration, currents))


@pytest.mark.parametrize("field", range(len(FIELDS)))
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_complete_constructor_and_empty_batch_refusal(field: int, bad: float) -> None:
    """No field escapes complete constructor or empty-batch admission."""
    configuration = list(DEFAULTS)
    configuration[field] = bad
    with pytest.raises(ValueError, match="invalid MAT state or configuration"):
        engine.MATNeuron(*configuration)
    drive = np.empty(0, dtype=np.float64)
    drive.setflags(write=False)
    with pytest.raises(ValueError, match="invalid MAT state or configuration"):
        _batch(tuple(configuration), drive)
    assert drive.size == 0 and not drive.flags.writeable


@pytest.mark.parametrize("dt", [0.001, 0.05, 0.5])
@pytest.mark.parametrize("current", [0.0, 0.5, 0.7])
def test_configured_complete_batch_and_class_match_public_python(dt: float, current: float) -> None:
    """Every configured dynamic trace, event and reset agrees with public Python."""
    configuration = (*PROFILE[:-1], dt)
    drive = np.resize(np.array([current, 0.0, 0.7, 0.3], dtype=np.float64), 256)
    before = drive.copy()
    drive.setflags(write=False)
    source = MATNeuron(*configuration)
    cell = engine.MATNeuron(*configuration)
    trace: list[tuple[float, float, float, float, int]] = []
    for sample in drive:
        event = source.step(float(sample))
        assert cell.step(float(sample)) == event
        state = cell.get_state()
        np.testing.assert_allclose(
            [state[key] for key in DYNAMICS],
            [getattr(source, key) for key in DYNAMICS],
            rtol=0,
            atol=2e-12,
        )
        trace.append((source.v, source.theta1, source.theta2, source.refractory_remaining, event))
    result = _batch(configuration, drive)
    for index, name in enumerate(("voltages", "theta1", "theta2", "refractory", "events")):
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
        ("refractory_final", source.refractory_remaining),
    ):
        assert result[name] == pytest.approx(value, abs=2e-12)
    np.testing.assert_array_equal(drive, before)
    assert not drive.flags.writeable
    assert cell.reset() is None
    source.reset()
    assert cell.get_state() == {key: getattr(source, key) for key in DYNAMICS}
    assert cell.step(0.7) == source.step(0.7)
    for key in DYNAMICS:
        assert cell.get_state()[key] == pytest.approx(getattr(source, key), abs=2e-12)


def test_empty_batch_retains_all_configured_dynamics() -> None:
    """Empty traces keep owning typed outputs and the complete initial dynamics."""
    result = _batch(PROFILE, np.empty(0, dtype=np.float64))
    assert set(result) == {
        "voltages",
        "theta1",
        "theta2",
        "refractory",
        "events",
        "v_final",
        "theta1_final",
        "theta2_final",
        "refractory_final",
    }
    for name in ("voltages", "theta1", "theta2", "refractory", "events"):
        output = result[name]
        assert output.shape == (0,)
        assert output.dtype == (np.int32 if name == "events" else np.float64)
        assert output.flags.owndata and output.flags.c_contiguous and output.flags.aligned
    assert (
        tuple(
            result[key] for key in ("v_final", "theta1_final", "theta2_final", "refractory_final")
        )
        == PROFILE[:4]
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1e308])
def test_invalid_transition_preserves_class_and_batch_input_before_retry(bad: float) -> None:
    """A refused step or late batch cannot poison a valid class or caller input."""
    cell = engine.MATNeuron(*PROFILE)
    control = engine.MATNeuron(*PROFILE)
    before = cell.get_state()
    with pytest.raises(ValueError):
        cell.step(bad)
    assert cell.get_state() == before
    assert cell.step(0.7) == control.step(0.7)
    assert cell.get_state() == control.get_state()
    drive = np.array([0.7, bad], dtype=np.float64)
    input_bits = drive.tobytes()
    drive.setflags(write=False)
    with pytest.raises(ValueError):
        _batch(PROFILE, drive)
    assert drive.tobytes() == input_bits and not drive.flags.writeable
    valid = _batch(PROFILE, np.array([0.7], dtype=np.float64))
    for key in DYNAMICS:
        final_key = "refractory_final" if key == "refractory_remaining" else key + "_final"
        assert valid[final_key] == pytest.approx(control.get_state()[key], abs=2e-12)
