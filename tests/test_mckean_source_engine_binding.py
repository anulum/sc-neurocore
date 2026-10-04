# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source McKean installed-engine contracts

"""Exercise the canonical McKean class and complete native batch boundary."""

from __future__ import annotations

import pickle
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.engine_requirement import require_engine

extension = require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.accel.mckean import simulate_mckean
from sc_neurocore.neurons.models.mckean import McKeanNeuron

CONFIG = (0.1, -0.05, 0.25, 1.0, 1.0, 0.01, 0.1)


def _batch(currents: NDArray[Any]) -> dict[str, Any]:
    """Call the complete installed source binding with a configured state."""
    return cast(dict[str, Any], extension.py_mckean_simulate(*CONFIG, currents))


def test_empty_batch_retains_configured_state_and_array_contract() -> None:
    """An empty drive returns typed empty traces and the supplied final state."""
    result = _batch(np.empty(0, dtype=np.float64))
    assert set(result) == {"voltages", "recovery", "events", "v_final", "w_final"}
    for key, dtype in (("voltages", np.float64), ("recovery", np.float64), ("events", np.int32)):
        trace = result[key]
        assert isinstance(trace, np.ndarray)
        assert trace.shape == (0,)
        assert trace.dtype == dtype
        assert trace.flags.c_contiguous
    assert (result["v_final"], result["w_final"]) == CONFIG[:2]


def test_readonly_batch_preserves_input_and_complete_source_trace() -> None:
    """A read-only float64 drive produces the enrolled complete source trace."""
    drive = np.array([0.5, 3.0, -0.2], dtype=np.float64)
    original = drive.copy()
    drive.setflags(write=False)
    result = _batch(drive)
    np.testing.assert_allclose(
        result["voltages"],
        [0.14281757504166667, 0.499531723323847, 0.5328167882673364],
        rtol=0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        result["recovery"],
        [-0.049878233062500006, -0.04956228996018745, -0.049045833771439146],
        rtol=0,
        atol=2e-15,
    )
    np.testing.assert_array_equal(result["events"], [0, 1, 0])
    for key in ("voltages", "recovery", "events"):
        assert result[key].shape == drive.shape
        assert result[key].flags.c_contiguous
        assert not np.shares_memory(result[key], drive)
    assert result["events"].dtype == np.int32
    assert result["v_final"] == result["voltages"][-1]
    assert result["w_final"] == result["recovery"][-1]
    np.testing.assert_array_equal(drive, original)
    assert drive.flags.writeable is False


@pytest.mark.parametrize("layout", ["float32", "matrix", "strided", "reversed", "misaligned"])
def test_native_batch_refuses_incompatible_dtype_shape_and_layout(layout: str) -> None:
    """Direct native callers receive the measured conversion/layout refusal."""
    arrays: dict[str, NDArray[Any]] = {
        "float32": np.zeros(4, dtype=np.float32),
        "matrix": np.zeros((2, 2), dtype=np.float64),
        "strided": np.arange(8, dtype=np.float64)[::2],
        "reversed": np.arange(4, dtype=np.float64)[::-1],
        "misaligned": np.frombuffer(bytearray(33), dtype=np.float64, count=4, offset=1),
    }
    drive = arrays[layout]
    before = drive.copy()
    with pytest.raises(TypeError) as captured:
        _batch(drive)
    expected = (
        "'ndarray' object is not an instance of 'ndarray'"
        if layout in {"float32", "matrix"}
        else "The given array is not contiguous or is misaligned."
    )
    assert str(captured.value) == expected
    np.testing.assert_array_equal(drive, before)


@pytest.mark.parametrize("drive", [[np.nan], [0.5, np.inf], [1e308]])
def test_native_batch_refuses_nonfinite_or_out_of_envelope_updates(drive: list[float]) -> None:
    """Invalid numeric candidates fail without changing the caller's array."""
    currents = np.array(drive, dtype=np.float64)
    before = currents.copy()
    with pytest.raises(ValueError) as captured:
        _batch(currents)
    expected = (
        "McKean RK4 candidate outside safety envelope"
        if drive[0] == 1e308
        else "invalid McKean state, configuration, or current"
    )
    assert str(captured.value) == expected
    np.testing.assert_array_equal(currents, before)


@pytest.mark.parametrize("kwargs", [{"a": 0.0}, {"mu": 0.25}, {"b": 0.0}, {"dt": 0.0}])
def test_configured_class_refuses_invalid_source_parameters(kwargs: dict[str, float]) -> None:
    """The Python constructor exposes the native source constraints."""
    with pytest.raises(ValueError) as captured:
        engine.McKeanNeuron(**kwargs)
    assert str(captured.value) == "invalid McKean state or configuration"


@pytest.mark.parametrize("current", [np.nan, np.inf, 1e308, "drive"])
def test_class_step_failure_preserves_state_and_next_transition(current: float | str) -> None:
    """Rejected numeric/conversion updates leave the next valid step intact."""
    neuron = engine.McKeanNeuron(*CONFIG)
    control = engine.McKeanNeuron(*CONFIG)
    before = neuron.get_state()
    with pytest.raises((TypeError, ValueError)) as captured:
        neuron.step(current)
    if isinstance(current, str):
        assert type(captured.value) is TypeError
        assert str(captured.value) == "must be real number, not str"
    else:
        assert type(captured.value) is ValueError
        expected = (
            "McKean RK4 candidate outside safety envelope"
            if current == 1e308
            else "invalid McKean state, configuration, or current"
        )
        assert str(captured.value) == expected
    assert neuron.get_state() == before
    assert neuron.step(0.5) == control.step(0.5)
    assert neuron.get_state() == control.get_state()


def test_state_copy_and_reset_preserve_source_configuration() -> None:
    """Returned state is a copy and reset keeps configured source parameters."""
    config = (0.1, -0.05, 0.3, 1.5, 2.0, 0.02, 0.2)
    neuron = engine.McKeanNeuron(*config)
    state = neuron.get_state()
    state["v"] = 99.0
    assert neuron.get_state() == {"v": config[0], "w": config[1]}
    neuron.step(3.0)
    neuron.reset()
    assert neuron.get_state() == {"v": 0.0, "w": 0.0}
    reference = McKeanNeuron(0.0, 0.0, *config[2:])
    assert neuron.step(0.5) == reference.step(0.5)
    assert neuron.get_state()["v"] == pytest.approx(reference.v, abs=2e-15)
    assert neuron.get_state()["w"] == pytest.approx(reference.w, abs=2e-15)


def test_instance_pickle_refusal_preserves_the_live_neuron() -> None:
    """Unsupported instance serialization has an explicit stable refusal."""
    neuron = engine.McKeanNeuron(*CONFIG)
    before = neuron.get_state()
    with pytest.raises(TypeError) as captured:
        pickle.dumps(neuron, protocol=4)
    assert str(captured.value) == (
        "cannot pickle 'sc_neurocore_engine.sc_neurocore_engine.McKeanNeuron' object"
    )
    assert neuron.get_state() == before


def test_public_rust_dispatch_matches_complete_python_reference() -> None:
    """The public dispatcher exercises the installed canonical batch function."""
    assert engine.py_mckean_simulate is extension.py_mckean_simulate
    drive = np.resize(np.array([0.0, 3.0, 0.0, -0.2]), 128)
    expected = simulate_mckean(drive, backend="python")
    actual = simulate_mckean(drive, backend="rust")
    np.testing.assert_array_equal(actual["events"], expected["events"])
    for key in ("voltages", "recovery"):
        actual_trace, expected_trace = actual[key], expected[key]
        assert isinstance(actual_trace, np.ndarray)
        assert isinstance(expected_trace, np.ndarray)
        np.testing.assert_allclose(actual_trace, expected_trace, rtol=0, atol=2e-12)
    assert actual["v_final"] == pytest.approx(expected["v_final"], abs=2e-12)
    assert actual["w_final"] == pytest.approx(expected["w_final"], abs=2e-12)
