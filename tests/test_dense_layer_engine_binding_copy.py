# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed dense copy refusal and recovery

"""Exercise real Python output and native input-copy allocation failures."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()

_PROGRAM = """
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import sc_neurocore_engine as engine
from sc_neurocore_engine import sc_neurocore_engine as native

origins = json.loads(sys.argv[1])
mode = sys.argv[2]
for module in (engine, native):
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve())
    assert str(path) == origins[module.__name__]['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == origins[module.__name__]['sha256']
assert engine.DenseLayer is native.DenseLayer
engine.set_num_threads(1)
pressure = sys.platform == 'linux'
previous = None
if pressure:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

def constrain(headroom):
    global previous
    if pressure:
        status = Path('/proc/self/status').read_text().splitlines()
        virtual = int(next(line.split()[1] for line in status
                           if line.startswith('VmSize:'))) * 1024
        previous = resource.getrlimit(resource.RLIMIT_AS)
        limit = virtual + headroom
        assert previous[1] == resource.RLIM_INFINITY or limit <= previous[1]
        resource.setrlimit(resource.RLIMIT_AS, (limit, previous[1]))

class LastRowLimit:
    '''Read finite caller-owned rows, applying pressure before the final row.'''
    def __init__(self, count):
        self.count = count
        self.final_row_read = False
        self.limit_on_final_row = True

    def __getitem__(self, index):
        if index >= self.count:
            raise IndexError(index)
        if index == self.count - 1:
            self.final_row_read = True
            if self.limit_on_final_row:
                constrain(1024**2)
        return (1.0,)

if mode in ('outer_output', 'owned_rows'):
    inputs, neurons, length = 1, 524288, 1
elif mode in ('inner_output', 'float_output', 'packed_outer'):
    inputs, neurons, length = 524288, 1, 1
else:
    assert mode == 'packed_inner'
    inputs, neurons, length = 1, 1, 33554432
layer = native.DenseLayer(inputs, neurons, length=length, seed=24301)
packed = None
rows = None
if mode.startswith('packed_'):
    packed = np.ones((inputs, (length + 63) // 64), dtype=np.uint64)
    if mode == 'packed_inner':
        packed.fill(np.iinfo(np.uint64).max)
    packed.setflags(write=False)
    packed_digest = hashlib.sha256(memoryview(packed)).hexdigest()
    operation = lambda: layer.forward_prepacked(packed)
elif mode == 'owned_rows':
    rows = LastRowLimit(neurons)
    operation = lambda: layer.set_weights(rows)
else:
    operation = layer.get_weights

if mode != 'owned_rows':
    constrain((6 if mode == 'float_output' else 1) * 1024**2)
refused = False
error_message = None
try:
    operation()
except MemoryError as error:
    refused = True
    error_message = str(error)
finally:
    if previous is not None:
        resource.setrlimit(resource.RLIMIT_AS, previous)
assert refused == pressure
if pressure and mode in ('owned_rows', 'packed_outer', 'packed_inner'):
    assert error_message == 'dense input cannot be allocated'
if rows is not None:
    assert rows.final_row_read
    rows.limit_on_final_row = False

control = native.DenseLayer(inputs, neurons, length=length, seed=24301)
expected = control.get_weights()
if pressure or mode != 'owned_rows':
    assert layer.get_weights() == expected
if packed is not None:
    assert hashlib.sha256(memoryview(packed)).hexdigest() == packed_digest
    assert not packed.flags.writeable
    assert layer.forward_prepacked(packed) == control.forward_prepacked(packed)
    assert layer.forward_prepacked_numpy(packed).tolist() == control.forward_prepacked(packed)
elif mode == 'owned_rows':
    layer.set_weights(rows)
    assert layer.get_weights() == [[1.0]] * neurons
    assert layer.forward_fast([1.0], seed=42) == [1.0] * neurons
else:
    recovered = layer.get_weights()
    assert recovered == expected
    recovered[0][0] = -1.0
    assert layer.get_weights() == expected

for module in (engine, native):
    path = Path(module.__file__).resolve()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == origins[module.__name__]['sha256']
print(json.dumps({'mode': mode, 'pressure_exercised': pressure, 'healthy_retry': True}))
"""


def _run_consumer(mode: str) -> None:
    """Require a restored isolated installed consumer after one copy operation.

    Parameters
    ----------
    mode
        Output representation or input-copy stage to exercise.
    """
    environment = os.environ.copy()
    environment.update(
        {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1"}
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _PROGRAM,
            json.dumps(installed_engine_origins()),
            mode,
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "mode": mode,
        "pressure_exercised": sys.platform == "linux",
        "healthy_retry": True,
    }


@pytest.mark.parametrize("mode", ["outer_output", "inner_output", "float_output"])
def test_dense_python_weight_copy_refusal_preserves_state_and_retry(mode: str) -> None:
    """Refuse large Python matrix/row/float copies and return independent retry rows.

    Parameters
    ----------
    mode
        Python output allocation to constrain without warming a large free buffer.
    """
    _run_consumer(mode)


@pytest.mark.parametrize("mode", ["owned_rows", "packed_outer", "packed_inner"])
def test_dense_native_input_copy_refusal_preserves_state_and_retry(mode: str) -> None:
    """Refuse owned rows and NumPy packed-row copies before mutating the layer.

    Parameters
    ----------
    mode
        Owned nested-sequence or packed-array allocation to exercise.
    """
    _run_consumer(mode)
