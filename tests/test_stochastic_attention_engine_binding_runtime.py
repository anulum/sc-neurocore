# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed attention worker admission and recovery

"""Exercise attention shape refusal, cold workers and same-instance recovery."""

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
@pytest.mark.parametrize(
    "method",
    ["forward", "forward_softmax", "forward_sc", "forward_multihead", "forward_multihead_softmax"],
)
def test_cold_worker_refusal_preserves_attention_inputs(surface: str, method: str) -> None:
    """Validate shapes before real worker refusal and retry the original inputs.

    Parameters
    ----------
    surface
        Installed extension or the maintained Python wrapper.
    method
        Linear, softmax, stochastic or multi-head attention entry point.
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
import sc_neurocore_engine.attention as facade
origins = cast(dict[str, dict[str, str]], json.loads(sys.argv[1]))
for module in (engine, native, facade):
    filename = module.__file__
    assert isinstance(filename, str)
    path = Path(filename).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve())
    assert str(path) == origins[module.__name__]['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == origins[module.__name__]['sha256']
assert engine.StochasticAttention is facade.StochasticAttention
surface, method = sys.argv[2:]
constructor = native.StochasticAttention if surface == 'native' else facade.StochasticAttention
attention = constructor(4)
q = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
k = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])
v = np.array([[1., 0., 1., 0.], [0., 1., 0., 1.], [1., 1., 0., 0.]])
for matrix in (q, k, v):
    matrix.setflags(write=False)
before = [matrix.tobytes() for matrix in (q, k, v)]
operation = getattr(attention, method)
arguments: dict[str, object] = {}
multihead = 'multihead' in method
if multihead:
    arguments['n_heads'] = 2
if method == 'forward_sc':
    arguments.update(length=65, seed=44257)
engine.set_num_threads(0)
try:
    operation(np.empty((0, 4)), k, v, **arguments)
except ValueError as caught:
    assert str(caught) == 'Q must not be an empty matrix.'
else:
    raise AssertionError('empty Q bypassed the matrix-input contract')
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
    error: MemoryError | None = None
    resource.setrlimit(resource.RLIMIT_AS, (limit, previous[1]))
    try:
        try:
            operation(q, k, v[:2], **arguments)
        except ValueError as caught:
            assert str(caught) == 'K/V row mismatch: k_rows=3, v_rows=2.'
        else:
            raise AssertionError('invalid K/V rows were accepted')
        if multihead:
            try:
                operation(q, k, v, n_heads=0)
            except ValueError as caught:
                assert str(caught) == 'n_heads must be > 0.'
            else:
                raise AssertionError('zero heads were accepted')
        if method == 'forward_sc':
            try:
                operation(q, k, v, length=0, seed=44257)
            except ValueError as caught:
                assert str(caught) == 'length must be > 0 for SC mode.'
            else:
                raise AssertionError('zero SC length was accepted')
        operation(q, k, v, **arguments)
    except MemoryError as caught:
        error = caught
    finally:
        resource.setrlimit(resource.RLIMIT_AS, previous)
    assert error is not None
    assert str(error) == 'parallel execution resources cannot be allocated'
    assert [matrix.tobytes() for matrix in (q, k, v)] == before
    assert all(not matrix.flags.writeable for matrix in (q, k, v))
actual = operation(q, k, v, **arguments)
assert isinstance(actual, list if surface == 'native' else np.ndarray)
if method == 'forward_sc':
    expected = v[:2]
else:
    heads = 2 if multihead else 1
    outputs = []
    for head in range(heads):
        q_head = q[:, head * (4 // heads):(head + 1) * (4 // heads)]
        k_head = k[:, head * (4 // heads):(head + 1) * (4 // heads)]
        v_head = v[:, head * (4 // heads):(head + 1) * (4 // heads)]
        scores = q_head @ k_head.T
        if 'softmax' in method:
            scores = scores / 2.0
            scores -= scores.max(axis=1, keepdims=True)
            weights = np.exp(scores)
            weights /= weights.sum(axis=1, keepdims=True)
        else:
            totals = scores.sum(axis=1, keepdims=True)
            totals[totals == 0.0] = 1.0
            weights = scores / totals
        outputs.append(weights @ v_head)
    expected = np.concatenate(outputs, axis=1)
np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
np.testing.assert_array_equal(operation(q, k, v, **arguments), actual)
assert [matrix.tobytes() for matrix in (q, k, v)] == before
assert all(not matrix.flags.writeable for matrix in (q, k, v))
try:
    engine.set_num_threads(2)
except ValueError as caught:
    assert str(caught) == 'Cannot set thread pool: The global thread pool has already been initialized.'
else:
    raise AssertionError('successful attention startup did not fix the global pool')
print(json.dumps({'surface': surface, 'method': method,
                  'pressure_exercised': pressure, 'input_recovery': True}))
"""
    origins = installed_engine_origins()
    wrapper = Path(origins["sc_neurocore_engine"]["path"]).parent / "attention.py"
    assert wrapper.is_relative_to(Path(sys.prefix).resolve())
    origins["sc_neurocore_engine.attention"] = {
        "path": str(wrapper),
        "sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
    }
    environment = os.environ.copy()
    environment.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    environment.pop("RAYON_NUM_THREADS", None)
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, json.dumps(origins), surface, method],
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
        "method": method,
        "pressure_exercised": sys.platform == "linux",
        "input_recovery": True,
    }
