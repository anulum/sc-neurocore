# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed differentiable dense worker admission and gradients

"""Exercise dense training constructor recovery and preserved gradient/cache behavior."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()


@pytest.mark.parametrize("surface", ["native", "facade"])
@pytest.mark.parametrize("surrogate", ["fast_sigmoid", "superspike", "arctan", "straight_through"])
def test_differentiable_dense_worker_refusal_and_gradient_recovery(
    surface: str, surrogate: str
) -> None:
    """Retry construction and compare complete gradients with the retained formula.

    Parameters
    ----------
    surface
        Installed native class or public NumPy facade.
    surrogate
        One of the four publicly parsed surrogate derivative families.
    """
    operation = f"{surface}:{surrogate}"
    program = """
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import cast
import numpy as np
import numpy.typing as npt
import sc_neurocore_engine as engine
import sc_neurocore_engine.sc_neurocore_engine as native
origins = cast(dict[str, dict[str, str]], json.loads(sys.argv[1]))
for module in (engine, native):
    filename = module.__file__
    assert isinstance(filename, str)
    path = Path(filename).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve())
    assert str(path) == origins[module.__name__]['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == origins[module.__name__]['sha256']
engine.set_num_threads(0)
pressure = sys.platform == 'linux'
operation = sys.argv[2]
surface, surrogate = operation.split(':')
import sc_neurocore_engine.grad as facade
filename = facade.__file__
assert isinstance(filename, str)
facade_path = Path(filename).resolve()
assert facade_path.is_relative_to(Path(sys.prefix).resolve())
facade_sha = hashlib.sha256(facade_path.read_bytes()).hexdigest()
if surface == 'facade':
    import sc_neurocore_engine.grad as facade
    filename = facade.__file__
    assert isinstance(filename, str) and Path(filename).resolve().is_relative_to(Path(sys.prefix).resolve())
    cls = facade.DifferentiableDenseLayer
else:
    cls = native.DifferentiableDenseLayer
if pressure:
    import resource
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    virtual = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                       if line.startswith('VmSize:'))) * 1024
    previous = resource.getrlimit(resource.RLIMIT_AS)
    limit = virtual + 1024**2
    assert previous[1] == resource.RLIM_INFINITY or limit <= previous[1]
    resource.setrlimit(resource.RLIMIT_AS, (limit, previous[1]))
error: MemoryError | None = None
try:
    try:
        cls(0, 0, length=0, surrogate='unknown')
    except ValueError as caught:
        assert str(caught) == "Unknown surrogate 'unknown'. Use one of: fast_sigmoid, superspike, arctan, straight_through."
    else:
        raise AssertionError('unknown surrogate was accepted')
    for dimensions, message in (
        ((0, 0, 0), 'bitstream length must be > 0'),
        ((0, 2, 65), 'n_inputs must be > 0'),
        ((4, 0, 65), 'n_neurons must be > 0'),
    ):
        try:
            cls(*dimensions, surrogate=surrogate)
        except ValueError as caught:
            assert str(caught) == message
        else:
            raise AssertionError('zero dense dimension was accepted')
    if pressure:
        cls(4, 2, length=65, seed=24301, surrogate=surrogate, k=2.)
except MemoryError as caught:
    error = caught
finally:
    if pressure:
        resource.setrlimit(resource.RLIMIT_AS, previous)
if pressure:
    assert error is not None
    assert str(error) == 'parallel execution resources cannot be allocated'
layer = cls(4, 2, length=65, seed=24301, surrogate=surrogate, k=2.)
control = cls(4, 2, length=65, seed=24301, surrogate=surrogate, k=2.)
weights = np.asarray(layer.weights if surface == 'facade' else layer.get_weights())
control_weights = np.asarray(control.weights if surface == 'facade' else control.get_weights())
np.testing.assert_array_equal(weights, control_weights)
try:
    layer.backward([1., 1.])
except ValueError as caught:
    assert str(caught) == 'backward() called before a valid forward() input cache.'
else:
    raise AssertionError('backward without forward cache was accepted')
inputs = [0., 1., 1., 0.]
out = np.asarray(layer.forward(inputs, seed=42))
np.testing.assert_array_equal(out, control.forward(inputs, seed=42))
try:
    layer.forward([1.], seed=99)
except ValueError:
    pass
else:
    raise AssertionError('wrong forward input shape was accepted')
try:
    layer.backward([1.])
except ValueError as caught:
    assert str(caught) == 'Expected grad_output length 2, got 1.'
else:
    raise AssertionError('wrong backward gradient shape was accepted')
grad_output = [1., -2.]
x = (out - 0.5).astype(np.float32)
k = np.float32(2.)
derivative: npt.NDArray[np.float32]
if surrogate == 'fast_sigmoid':
    denom = np.float32(1.) + k * np.abs(x)
    derivative = np.asarray(np.float32(1.) / (np.float32(2.) * k * denom * denom), dtype=np.float32)
elif surrogate == 'superspike':
    denom = np.float32(1.) + k * np.abs(x)
    derivative = np.asarray(np.float32(1.) / (denom * denom), dtype=np.float32)
elif surrogate == 'arctan':
    derivative = np.asarray(np.float32(1.) / (np.float32(1.) + (k * x) ** 2), dtype=np.float32)
else:
    derivative = (np.abs(x) < np.float32(0.5)).astype(np.float32)
local = np.array(grad_output) * derivative.astype(np.float64)
grad_input, grad_weights = layer.backward(grad_output)
np.testing.assert_allclose(grad_input, local @ weights, rtol=1e-14, atol=1e-15)
np.testing.assert_allclose(grad_weights, local[:, None] * np.array(inputs)[None, :], rtol=1e-14, atol=1e-15)
np.testing.assert_array_equal(layer.backward(grad_output)[0], grad_input)
for bad in ([[1., 2., 3., 4.]], [[1.], [2.]]):
    try:
        layer.update_weights(bad, lr=0.1)
    except ValueError:
        pass
    else:
        raise AssertionError('wrong update gradient shape was accepted')
np.testing.assert_array_equal(layer.weights if surface == 'facade' else layer.get_weights(), weights)
layer.update_weights(np.asarray(grad_weights).tolist(), lr=0.1)
expected_weights = np.clip(weights - 0.1 * np.asarray(grad_weights), 0., 1.)
np.testing.assert_allclose(layer.weights if surface == 'facade' else layer.get_weights(), expected_weights,
                           rtol=0., atol=0.)
layer.clear_cache()
try:
    layer.backward(grad_output)
except ValueError as caught:
    assert str(caught) == 'backward() called before a valid forward() input cache.'
else:
    raise AssertionError('cleared cache was reused')
assert inputs == [0., 1., 1., 0.] and grad_output == [1., -2.]
assert Path(str(facade.__file__)).resolve() == facade_path
assert hashlib.sha256(facade_path.read_bytes()).hexdigest() == facade_sha
try:
    engine.set_num_threads(2)
except ValueError as caught:
    assert str(caught) == 'Cannot set thread pool: The global thread pool has already been initialized.'
else:
    raise AssertionError('healthy parallel execution did not fix the shared pool')
print(json.dumps({'operation': operation, 'pressure_exercised': pressure, 'recovery': True}))
"""
    environment = os.environ.copy()
    environment.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    environment.pop("RAYON_NUM_THREADS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            program,
            json.dumps(installed_engine_origins()),
            operation,
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=40,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "operation": operation,
        "pressure_exercised": sys.platform == "linux",
        "recovery": True,
    }
