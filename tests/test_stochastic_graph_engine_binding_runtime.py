# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed graph worker admission and recovery

"""Exercise dense and CSR graph worker refusal without warming serial SC paths."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()


@pytest.mark.parametrize("surface", ["native", "facade"])
@pytest.mark.parametrize("storage", ["dense", "sparse", "auto"])
def test_graph_worker_refusal_preserves_inputs_and_serial_mode(surface: str, storage: str) -> None:
    """Keep SC execution serial and recover the original graph after worker refusal.

    Parameters
    ----------
    surface
        Installed extension or the maintained Python wrapper.
    storage
        Dense, explicit CSR or automatically selected CSR adjacency.
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
import sc_neurocore_engine.graphs as facade
origins = cast(dict[str, dict[str, str]], json.loads(sys.argv[1]))
for module in (engine, native, facade):
    filename = module.__file__
    assert isinstance(filename, str)
    path = Path(filename).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve())
    assert str(path) == origins[module.__name__]['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == origins[module.__name__]['sha256']
assert engine.StochasticGraphLayer is facade.StochasticGraphLayer
surface, storage = sys.argv[2:]
constructor = native.StochasticGraphLayer if surface == 'native' else facade.StochasticGraphLayer
adjacency = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
features = np.array([[1., 0.], [0., 1.], [1., 1.]])
adjacency.setflags(write=False)
features.setflags(write=False)
before = features.tobytes()
adjacency_before = adjacency.tobytes()
if storage == 'dense':
    layer = constructor(adjacency, 2, seed=42)
elif storage == 'sparse':
    layer = constructor.from_sparse([0, 1, 2, 3], [1, 2, 0], [1., 1., 1.], 3, 2, seed=42)
else:
    layer = constructor.from_dense_auto(adjacency, 2, seed=42, density_threshold=0.9)
assert layer.is_sparse() == (storage != 'dense')
weights = np.asarray(layer.get_weights() if surface == 'native' else layer.weights).reshape(2, 2)
weights_before = weights.tobytes()
engine.set_num_threads(0)
try:
    constructor(np.empty((0, 0)), 2, seed=42)
except ValueError as caught:
    assert str(caught) == 'adj_matrix must not be an empty matrix.'
else:
    raise AssertionError('empty adjacency bypassed the matrix-input contract')
engine.set_num_threads(0)
serial = np.asarray(layer.forward_sc(features, length=65, seed=44257)).copy()
pressure = sys.platform == 'linux'
if pressure:
    import resource
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    virtual = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                       if line.startswith('VmSize:'))) * 1024
    previous = resource.getrlimit(resource.RLIMIT_AS)
    limit = virtual + 1024**2
    assert previous[1] == resource.RLIM_INFINITY or limit <= previous[1]
    error: MemoryError | None = None
    resource.setrlimit(resource.RLIMIT_AS, (limit, previous[1]))
    try:
        try:
            layer.forward(features[:2])
        except ValueError as caught:
            assert str(caught) == 'Expected node_features shape (3, 2), got (2, 2).'
        else:
            raise AssertionError('wrong feature shape was accepted')
        layer.forward(features)
    except MemoryError as caught:
        error = caught
    finally:
        resource.setrlimit(resource.RLIMIT_AS, previous)
    assert error is not None
    assert str(error) == 'parallel execution resources cannot be allocated'
    assert features.tobytes() == before and not features.flags.writeable
    actual_weights = layer.get_weights() if surface == 'native' else layer.weights
    assert np.asarray(actual_weights).reshape(2, 2).tobytes() == weights_before
actual = layer.forward(features)
assert isinstance(actual, list if surface == 'native' else np.ndarray)
degrees = adjacency.sum(axis=1, keepdims=True)
aggregate = (adjacency @ features) / degrees
expected = np.tanh(aggregate @ weights)
np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
np.testing.assert_array_equal(layer.forward_sc(features, length=65, seed=44257), serial)
assert features.tobytes() == before and not features.flags.writeable
assert adjacency.tobytes() == adjacency_before and not adjacency.flags.writeable
actual_weights = layer.get_weights() if surface == 'native' else layer.weights
assert np.asarray(actual_weights).reshape(2, 2).tobytes() == weights_before
try:
    engine.set_num_threads(2)
except ValueError as caught:
    assert str(caught) == 'Cannot set thread pool: The global thread pool has already been initialized.'
else:
    raise AssertionError('successful graph startup did not fix the global pool')
print(json.dumps({'surface': surface, 'storage': storage,
                  'pressure_exercised': pressure, 'input_recovery': True}))
"""
    origins = installed_engine_origins()
    wrapper = Path(origins["sc_neurocore_engine"]["path"]).parent / "graphs.py"
    assert wrapper.is_relative_to(Path(sys.prefix).resolve())
    origins["sc_neurocore_engine.graphs"] = {
        "path": str(wrapper),
        "sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
    }
    environment = os.environ.copy()
    environment.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    environment.pop("RAYON_NUM_THREADS", None)
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, json.dumps(origins), surface, storage],
        capture_output=True,
        text=True,
        env=environment,
        timeout=40,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    observation = json.loads(result.stdout.splitlines()[-1])
    assert observation == {
        "surface": surface,
        "storage": storage,
        "pressure_exercised": sys.platform == "linux",
        "input_recovery": True,
    }
