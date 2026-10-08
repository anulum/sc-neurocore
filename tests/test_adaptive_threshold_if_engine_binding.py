# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Configured adaptive-threshold binding contracts

"""Exercise configured batch extraction, numerical errors and allocation recovery."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
import subprocess
import sys
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel.adaptive_threshold_if import simulate_python
from sc_neurocore.neurons.models.adaptive_threshold_if import AdaptiveThresholdIFNeuron
from tests.engine_requirement import installed_engine_origins, require_engine

_NATIVE = require_engine()
_PARAMETERS = (-63.5, -52.5, -68.0, -67.0, -49.0, 4.5, 8.0, 42.0, 0.05)
_FUNCTION = cast(Callable[..., dict[str, object]], _NATIVE.py_adaptive_threshold_if_simulate)


class _ScalarNeuron(Protocol):
    def step(self, current: float) -> int: ...

    def get_state(self) -> dict[str, float]: ...

    def reset(self) -> None: ...


_SCALAR = cast(Callable[[], _ScalarNeuron], _NATIVE.AdaptiveThresholdIFNeuron)


@pytest.mark.parametrize("current", [float("nan"), float("inf"), -float("inf")])
def test_default_scalar_nonfinite_refusal_matches_python_and_is_atomic(current: float) -> None:
    """Refuse non-finite current before mutation and recover the next scalar step.

    Parameters
    ----------
    current
        NaN or either signed infinity supplied to the actual default class.
    """
    neuron, golden = _SCALAR(), AdaptiveThresholdIFNeuron()
    before = neuron.get_state()
    with pytest.raises(ValueError) as python_refusal:
        golden.step(current)
    with pytest.raises(ValueError) as native_refusal:
        neuron.step(current)
    assert str(native_refusal.value) == str(python_refusal.value) == "current must be finite"
    assert neuron.get_state() == before
    assert neuron.step(1.25) == golden.step(1.25)
    np.testing.assert_allclose(
        list(neuron.get_state().values()), [golden.v, golden.theta], rtol=0, atol=1e-12
    )


def test_default_scalar_candidate_refusal_is_atomic_and_matches_python() -> None:
    """Report a finite-entry overflow without replacing it with a silent zero event."""
    maximum = float(np.finfo(np.float64).max)
    neuron, golden = _SCALAR(), AdaptiveThresholdIFNeuron()
    assert neuron.step(-maximum) == golden.step(-maximum)
    before = neuron.get_state()
    with pytest.raises(FloatingPointError) as python_refusal:
        golden.step(maximum)
    with pytest.raises(FloatingPointError) as native_refusal:
        neuron.step(maximum)
    assert str(native_refusal.value) == str(python_refusal.value)
    assert str(native_refusal.value) == "exact relaxation candidate must be finite"
    assert neuron.get_state() == before
    neuron.reset()
    golden.reset()
    assert neuron.get_state() == {"v": golden.v, "theta": golden.theta}
    assert neuron.step(1.25) == golden.step(1.25)


def test_default_scalar_full_trace_state_and_reset_match_python() -> None:
    """Retain every post-step state, event and detached mapping through reset."""
    neuron, golden = _SCALAR(), AdaptiveThresholdIFNeuron()
    for current in [0.0, 4.0, 22.0, 50.0, -1.0, 120.0] * 32:
        assert neuron.step(current) == golden.step(current)
        state = neuron.get_state()
        np.testing.assert_allclose(
            list(state.values()), [golden.v, golden.theta], rtol=0, atol=1e-12
        )
        state.clear()
        assert tuple(neuron.get_state()) == ("v", "theta")
    neuron.reset()
    golden.reset()
    assert neuron.get_state() == {"v": golden.v, "theta": golden.theta}


def test_default_scalar_persistence_refuses_without_changing_state() -> None:
    """Resolve both native globals and preserve an advanced instance on every refusal."""
    import copy
    import pickle

    neuron = _SCALAR()
    neuron.step(22.0)
    before = neuron.get_state()
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        assert pickle.loads(pickle.dumps(_SCALAR, protocol)) is _SCALAR
        assert pickle.loads(pickle.dumps(_FUNCTION, protocol)) is _FUNCTION
        with pytest.raises(TypeError):
            pickle.dumps(neuron, protocol)
        assert neuron.get_state() == before
    for copier in (copy.copy, copy.deepcopy):
        with pytest.raises(TypeError):
            copier(neuron)
        assert neuron.get_state() == before


@pytest.mark.parametrize("steps", [0, 1, 128, 1024])
def test_configured_readonly_batch_matches_python_and_owns_results(steps: int) -> None:
    """Compare every returned field and prevent output edits from changing the drive."""
    index = np.arange(steps, dtype=np.float64)
    drive = 22.0 + 6.0 * np.sin(index * 0.037) + 1.5 * np.cos(index * 0.011)
    drive.setflags(write=False)
    before = drive.tobytes()
    expected = simulate_python(*_PARAMETERS, drive)
    actual = _FUNCTION(*_PARAMETERS, drive)
    assert tuple(actual) == ("v", "theta", "spikes", "v_final", "theta_final", "spike_count")
    for key in ("v", "theta", "spikes"):
        trace = actual[key]
        assert isinstance(trace, np.ndarray)
        assert trace.shape == (steps,) and trace.dtype == np.float64
        assert trace.flags.c_contiguous and trace.flags.writeable
        assert not np.shares_memory(trace, drive)
        np.testing.assert_allclose(trace, expected[key], rtol=0.0, atol=1e-12)
    assert actual["v_final"] == pytest.approx(expected["v_final"], abs=1e-12)
    assert actual["theta_final"] == pytest.approx(expected["theta_final"], abs=1e-12)
    assert actual["spike_count"] == expected["spike_count"]
    for key in ("v", "theta", "spikes"):
        cast(npt.NDArray[np.float64], actual[key]).fill(123.0)
    assert drive.tobytes() == before and not drive.flags.writeable
    repeated = _FUNCTION(*_PARAMETERS, drive)
    for key in ("v", "theta", "spikes"):
        repeated_trace = repeated[key]
        assert isinstance(repeated_trace, np.ndarray)
        np.testing.assert_allclose(repeated_trace, expected[key], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("kind", ["float32", "int32", "rank0", "rank2", "byteswapped"])
def test_array_dtype_rank_and_byteorder_are_not_silently_converted(kind: str) -> None:
    """Reject incompatible NumPy storage without changing the caller's bytes."""
    values: dict[str, npt.NDArray[np.generic]] = {
        "float32": np.ones(3, dtype=np.float32),
        "int32": np.ones(3, dtype=np.int32),
        "rank0": np.array(1.0),
        "rank2": np.ones((2, 3)),
        "byteswapped": np.ones(3, dtype=np.dtype(np.float64).newbyteorder("S")),
    }
    drive = values[kind]
    before = drive.tobytes()
    with pytest.raises(TypeError, match="is not an instance of 'ndarray'"):
        _FUNCTION(*_PARAMETERS, drive)
    assert drive.tobytes() == before


@pytest.mark.parametrize("kind", ["stride", "reverse", "unaligned"])
def test_array_layout_refusal_preserves_input_and_recovers(kind: str) -> None:
    """Reject strided or misaligned buffers before numerical execution."""
    storage = bytearray(25)
    candidates = {
        "stride": np.arange(6.0)[::2],
        "reverse": np.arange(3.0)[::-1],
        "unaligned": np.ndarray((3,), dtype=np.float64, buffer=storage, offset=1),
    }
    drive = candidates[kind]
    drive.setflags(write=False)
    before = drive.tobytes()
    with pytest.raises(TypeError) as refused:
        _FUNCTION(*_PARAMETERS, drive)
    assert str(refused.value) == "The given array is not contiguous or is misaligned."
    assert drive.tobytes() == before and not drive.flags.writeable
    actual = _FUNCTION(*_PARAMETERS, np.ascontiguousarray(drive).copy())
    assert cast(npt.NDArray[np.float64], actual["v"]).shape == (3,)


@pytest.mark.parametrize("parameter", range(9))
def test_nonfinite_configuration_is_rejected_even_for_empty_drive(parameter: int) -> None:
    """Validate each configured scalar independently of the batch length."""
    parameters = list(_PARAMETERS)
    parameters[parameter] = float("nan")
    with pytest.raises(ValueError) as refused:
        _FUNCTION(*parameters, np.empty(0))
    assert str(refused.value) == "adaptive-threshold state and parameters must be finite"


@pytest.mark.parametrize("parameter,value", [(5, -1.0), (6, 0.0), (7, 0.0), (8, 0.0), (4, -68.0)])
def test_invalid_scale_refusal_is_an_authored_configuration_error(
    parameter: int, value: float
) -> None:
    """Refuse invalid magnitude and rest/reset ordering before allocating traces."""
    parameters = list(_PARAMETERS)
    parameters[parameter] = value
    with pytest.raises(ValueError) as refused:
        _FUNCTION(*parameters, np.empty(0))
    assert str(refused.value) == (
        "adaptive-threshold delta_theta must be non-negative, tau_m/tau_theta/dt must be "
        "positive, and theta_rest must exceed v_rest and v_reset"
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_drive_is_refused_without_mutating_caller_storage(value: float) -> None:
    """Validate the whole drive and keep the next valid call usable."""
    drive = np.array([1.0, value, 2.0])
    drive.setflags(write=False)
    before = drive.tobytes()
    with pytest.raises(ValueError) as refused:
        _FUNCTION(*_PARAMETERS, drive)
    assert str(refused.value) == "adaptive-threshold input must contain only finite values"
    assert drive.tobytes() == before and not drive.flags.writeable
    assert _FUNCTION(*_PARAMETERS, np.empty(0))["v_final"] == _PARAMETERS[0]


def test_nonfinite_numerical_candidate_has_a_distinct_error_and_no_input_mutation() -> None:
    """Report overflow of a finite configured candidate as FloatingPointError."""
    maximum = np.finfo(np.float64).max
    parameters = (maximum, -1e308, -maximum, -maximum, -1e307, 0.0, 8.0, 42.0, 0.05)
    drive = np.zeros(1)
    drive.setflags(write=False)
    with pytest.raises(FloatingPointError) as refused:
        _FUNCTION(*parameters, drive)
    assert str(refused.value) == (
        "adaptive-threshold exact-relaxation candidate must remain finite"
    )
    np.testing.assert_array_equal(drive, [0.0])
    assert not drive.flags.writeable


def test_installed_trace_allocation_refuses_real_pressure_and_recovers() -> None:
    """Keep the native process alive when its three trace buffers cannot be reserved."""
    program = r"""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import sc_neurocore_engine as engine
import sc_neurocore_engine.sc_neurocore_engine as native

origins = json.loads(sys.argv[1])
for module in (engine, native):
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve())
    assert str(path) == origins[module.__name__]['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == origins[module.__name__]['sha256']
parameters = (-63.5, -52.5, -68., -67., -49., 4.5, 8., 42., .05)
drive = np.full(300_000, 1.25)
drive.setflags(write=False)
before = hashlib.sha256(memoryview(drive)).hexdigest()
pressure = sys.platform == 'linux'
if pressure:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    previous = resource.getrlimit(resource.RLIMIT_AS)
    virtual = int(Path('/proc/self/statm').read_text().split()[0]) * os.sysconf('SC_PAGE_SIZE')
    budget = int(sys.argv[2]) * 1024**2
    limit = virtual + budget
    assert previous[1] == resource.RLIM_INFINITY or limit <= previous[1]
    print(json.dumps({'phase': 'entering', 'virtual_bytes': virtual,
                      'headroom_bytes': budget, 'trace_bytes': drive.nbytes}), flush=True)
    resource.setrlimit(resource.RLIMIT_AS, (limit, previous[1]))
    try:
        engine.py_adaptive_threshold_if_simulate(*parameters, drive)
    except MemoryError as refused:
        assert str(refused) == 'adaptive-threshold trace buffers cannot be allocated'
    else:
        raise AssertionError('native trace allocation must refuse actual pressure')
    finally:
        resource.setrlimit(resource.RLIMIT_AS, previous)
    assert resource.getrlimit(resource.RLIMIT_AS) == previous
assert hashlib.sha256(memoryview(drive)).hexdigest() == before
assert not drive.flags.writeable
result = engine.py_adaptive_threshold_if_simulate(*parameters, drive[:3])
assert result['v'].shape == result['theta'].shape == result['spikes'].shape == (3,)
expected_v, expected_theta = parameters[:2]
for index in range(3):
    expected_v = parameters[2] + 1.25 + (expected_v - parameters[2] - 1.25) * np.exp(-.05/8.)
    expected_theta = parameters[4] + (expected_theta - parameters[4]) * np.exp(-.05/42.)
    np.testing.assert_allclose(result['v'][index], expected_v, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result['theta'][index], expected_theta, rtol=0, atol=1e-12)
np.testing.assert_array_equal(result['spikes'], np.zeros(3))
print(json.dumps({'pressure': pressure, 'inputs_unchanged': True, 'recovery': True}), flush=True)
"""
    environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    environment.pop("PYTHONPATH", None)
    budgets = (1, 3, 5) if sys.platform == "linux" else (0,)
    for budget in budgets:
        process = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                program,
                json.dumps(installed_engine_origins()),
                str(budget),
            ],
            cwd=os.environ.get("TMPDIR", os.getcwd()),
            env=environment,
            capture_output=True,
            text=True,
            timeout=40,
            check=False,
        )
        assert process.returncode == 0, (budget, process.returncode, process.stdout, process.stderr)
        result = json.loads(process.stdout.splitlines()[-1])
        assert result == {
            "pressure": sys.platform == "linux",
            "inputs_unchanged": True,
            "recovery": True,
        }
