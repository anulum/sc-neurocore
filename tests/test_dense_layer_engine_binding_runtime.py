# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed dense constructor worker refusal and recovery

"""Exercise dense constructor admission and retained forward-kernel thresholds."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()


@pytest.mark.parametrize(("n_inputs", "n_neurons"), [(4, 2), (16, 8), (128, 8)])
def test_dense_constructor_worker_refusal_and_forward_recovery(
    n_inputs: int, n_neurons: int
) -> None:
    """Retry cold dense construction and retain every public forward representation.

    Parameters
    ----------
    n_inputs
        Feature count below or at the parallel encoding threshold.
    n_neurons
        Output count below or at the parallel neuron threshold.
    """
    program = """
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import cast
import numpy as np
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
assert engine.DenseLayer is native.DenseLayer
n_inputs, n_neurons = map(int, sys.argv[2:])
engine.set_num_threads(0)
pressure = sys.platform == 'linux'
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
    for dimensions, message in (
        ((0, 0, 0), 'bitstream length must be > 0'),
        ((0, n_neurons, 65), 'n_inputs must be > 0'),
        ((n_inputs, 0, 65), 'n_neurons must be > 0'),
    ):
        try:
            native.DenseLayer(*dimensions, seed=24301)
        except ValueError as caught:
            assert str(caught) == message
        else:
            raise AssertionError('zero dense dimension was accepted')
    if pressure:
        native.DenseLayer(n_inputs, n_neurons, length=65, seed=24301)
except MemoryError as caught:
    error = caught
finally:
    if pressure:
        resource.setrlimit(resource.RLIMIT_AS, previous)
if pressure:
    assert error is not None
    assert str(error) == 'parallel execution resources cannot be allocated'
layer = native.DenseLayer(n_inputs, n_neurons, length=65, seed=24301)
control = native.DenseLayer(n_inputs, n_neurons, length=65, seed=24301)
np.testing.assert_array_equal(layer.get_weights(), control.get_weights())
weights = ((np.arange(n_neurons)[:, None] + np.arange(n_inputs)[None, :]) % 2).astype(np.float64)
inputs = (np.arange(n_inputs) % 2).astype(np.float64)
inputs.setflags(write=False)
before = inputs.tobytes()
layer.set_weights(weights.tolist())
expected = weights @ inputs
np.testing.assert_array_equal(layer.forward(inputs.tolist(), seed=42), expected)
np.testing.assert_array_equal(layer.forward_fast(inputs.tolist(), seed=42), expected)
out = layer.forward_numpy(inputs, seed=42)
assert out.dtype == np.float64 and out.flags.c_contiguous
np.testing.assert_array_equal(out, expected)
packed = np.zeros((n_inputs, 2), dtype=np.uint64)
packed[inputs == 1., 0] = np.uint64(2**64 - 1)
packed[inputs == 1., 1] = np.uint64(1)
packed.setflags(write=False)
packed_before = packed.tobytes()
np.testing.assert_array_equal(layer.forward_prepacked(packed), expected)
np.testing.assert_array_equal(layer.forward_prepacked(packed.tolist()), expected)
np.testing.assert_array_equal(layer.forward_prepacked_numpy(packed), expected)
batch = np.stack([inputs, 1.0 - inputs])
batch.setflags(write=False)
batch_before = batch.tobytes()
np.testing.assert_array_equal(layer.forward_batch_numpy(batch, seed=42), batch @ weights.T)
empty = layer.forward_batch_numpy(np.empty((0, n_inputs)), seed=42)
assert empty.shape == (0, n_neurons) and empty.dtype == np.float64
layer.refresh_packed_weights()
np.testing.assert_array_equal(layer.get_weights(), weights)
np.testing.assert_array_equal(layer.forward_fast(inputs.tolist(), seed=42), expected)
assert inputs.tobytes() == before and not inputs.flags.writeable
assert packed.tobytes() == packed_before and not packed.flags.writeable
assert batch.tobytes() == batch_before and not batch.flags.writeable
try:
    engine.set_num_threads(2)
except ValueError as caught:
    assert str(caught) == 'Cannot set thread pool: The global thread pool has already been initialized.'
else:
    raise AssertionError('successful dense creation did not fix the global pool')
print(json.dumps({'n_inputs': n_inputs, 'n_neurons': n_neurons,
                  'pressure_exercised': pressure, 'constructor_recovery': True}))
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
            str(n_inputs),
            str(n_neurons),
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=40,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    observation = json.loads(result.stdout.splitlines()[-1])
    assert observation == {
        "n_inputs": n_inputs,
        "n_neurons": n_neurons,
        "pressure_exercised": sys.platform == "linux",
        "constructor_recovery": True,
    }
