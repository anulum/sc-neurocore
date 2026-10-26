# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed dense allocation refusal and recovery

"""Exercise dense capacity and allocation refusals in isolated installed consumers."""

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
from typing import Generic, TypeVar
import numpy as np
import sc_neurocore_engine as engine
import sc_neurocore_engine.sc_neurocore_engine as native

origins = json.loads(sys.argv[1])
mode = sys.argv[2]
for module in (engine, native):
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve())
    assert str(path) == origins[module.__name__]['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == origins[module.__name__]['sha256']
assert engine.DenseLayer is native.DenseLayer
engine.set_num_threads(1)
if sys.platform == 'linux':
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

if mode == 'capacity':
    dimensions = json.loads(sys.argv[3])
    try:
        native.DenseLayer(*dimensions, seed=24301)
    except MemoryError as error:
        assert str(error) == 'dense storage cannot be allocated'
    else:
        raise AssertionError('unrepresentable dense storage was accepted')
elif mode in ('flat_hint', 'outer_hint', 'inner_hint'):
    Item = TypeVar('Item')
    class HugeSequence(Generic[Item]):
        '''Provide finite actual items with an unrepresentable reservation hint.'''
        def __init__(self, items: tuple[Item, ...] | list[Item]) -> None:
            '''Keep caller-owned items for sequence-protocol reads.

            Parameters
            ----------
            items
                Finite actual scalars or matrix rows.
            '''
            self.items = items
        def __len__(self) -> int:
            '''Report a length that cannot fit a native scalar or row vector.'''
            return sys.maxsize
        def __getitem__(self, index: int) -> Item:
            '''Read one actual item or end iteration through IndexError.

            Parameters
            ----------
            index
                Actual element offset requested by the sequence iterator.

            Returns
            -------
            Item
                Caller-owned scalar or matrix row at that offset.
            '''
            return self.items[index]
    layer = native.DenseLayer(2, 2, length=65, seed=24301)
    layer.set_weights([[1.0, 0.0], [0.0, 1.0]])
    if mode == 'flat_hint':
        values = HugeSequence((1.0, 0.0))
        assert layer.forward(values, seed=42) == [1.0, 0.0]
        assert layer.forward_fast(values, seed=42) == [1.0, 0.0]
        assert layer.get_weights() == [[1.0, 0.0], [0.0, 1.0]]
    else:
        rows = [[0.0, 1.0], [1.0, 0.0]]
        values = HugeSequence(rows) if mode == 'outer_hint' else [HugeSequence(row) for row in rows]
        layer.set_weights(values)
        assert layer.get_weights() == rows
        assert layer.forward_fast([1.0, 0.0], seed=42) == [0.0, 1.0]
        layer.refresh_packed_weights()
        assert layer.get_weights() == rows
        assert layer.forward_fast([1.0, 0.0], seed=42) == [0.0, 1.0]
else:
    pressure = sys.platform == 'linux'
    n_inputs = 1
    if mode in ('set_weights', 'refresh_packed_weights'):
        neurons, length = 8192, 2048
    elif mode == 'forward_batch_numpy':
        neurons, length = 8192, 1
    elif mode in ('forward_batch_numpy_sample', 'forward_numpy_packed'):
        n_inputs, neurons, length = 8192, 1, 2048
    elif mode == 'constructor':
        neurons, length = 4096, 4096
    else:
        neurons, length = 524288, 1
    layer = native.DenseLayer(n_inputs, neurons, length=length, seed=24301)
    replacement = [[1.0] * n_inputs] * neurons
    if mode not in ('set_weights', 'refresh_packed_weights', 'constructor'):
        layer.set_weights(replacement)
    if mode in ('set_weights', 'refresh_packed_weights'):
        original_weights = layer.get_weights()
        original_forward = layer.forward_fast([1.0], seed=42)
    values = np.ones(n_inputs, dtype=np.float64)
    packed = np.ones((1, 1), dtype=np.uint64)
    samples = 1 if mode in ('forward_batch_numpy_sample', 'forward_numpy_packed') else 1024
    batch = np.ones((samples, n_inputs), dtype=np.float64)
    for array in (values, packed, batch):
        array.setflags(write=False)
    before = [array.tobytes() for array in (values, packed, batch)]
    methods = {
        'constructor': lambda: native.DenseLayer(3, neurons, length=length, seed=24301),
        'get_weights': layer.get_weights,
        'set_weights': lambda: layer.set_weights(replacement),
        'refresh_packed_weights': layer.refresh_packed_weights,
        'forward': lambda: layer.forward([1.0], seed=42),
        'forward_fast': lambda: layer.forward_fast([1.0], seed=42),
        'forward_numpy': lambda: layer.forward_numpy(values, seed=42),
        'forward_numpy_packed': lambda: layer.forward_numpy(values, seed=42),
        'forward_prepacked_list': lambda: layer.forward_prepacked([[1]]),
        'forward_prepacked_array': lambda: layer.forward_prepacked(packed),
        'forward_prepacked_numpy': lambda: layer.forward_prepacked_numpy(packed),
        'forward_batch_numpy': lambda: layer.forward_batch_numpy(batch, seed=42),
        'forward_batch_numpy_sample': lambda: layer.forward_batch_numpy(batch, seed=42),
    }
    retained_outputs = []
    if pressure and mode in ('forward_numpy', 'forward_prepacked_numpy', 'forward_batch_numpy'):
        for attempt in range(32):
            virtual_before = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                                      if line.startswith('VmSize:'))) * 1024
            completed = methods[mode]()
            assert np.all(completed == float(n_inputs))
            retained_outputs.append(completed)
            virtual_after = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                                     if line.startswith('VmSize:'))) * 1024
            assert sum(output.nbytes for output in retained_outputs) <= 128 * 1024**2
            if virtual_after - virtual_before >= completed.nbytes:
                break
        else:
            raise AssertionError('no allocator growth established before pressure')
    if pressure:
        virtual = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                           if line.startswith('VmSize:'))) * 1024
        previous = resource.getrlimit(resource.RLIMIT_AS)
        limit = virtual + 1024**2
        assert previous[1] == resource.RLIM_INFINITY or limit <= previous[1]
        resource.setrlimit(resource.RLIMIT_AS, (limit, previous[1]))
    refused = False
    try:
        methods[mode]()
    except MemoryError as error:
        assert pressure
        if mode == 'forward_numpy_packed':
            assert str(error) == 'dense storage cannot be allocated'
        refused = True
    finally:
        if pressure:
            resource.setrlimit(resource.RLIMIT_AS, previous)
    assert refused == pressure
    for array, saved in zip((values, packed, batch), before):
        assert array.tobytes() == saved and not array.flags.writeable
    if mode in ('set_weights', 'refresh_packed_weights') and pressure:
        assert layer.get_weights() == original_weights
        assert layer.forward_fast([1.0], seed=42) == original_forward
    elif mode not in ('set_weights', 'refresh_packed_weights', 'constructor'):
        recovered = methods[mode]()
        assert np.all(np.asarray(recovered) == float(n_inputs))

healthy = native.DenseLayer(2, 2, length=65, seed=24301)
healthy.set_weights([[1.0, 0.0], [0.0, 1.0]])
assert healthy.forward_fast([1.0, 0.0], seed=42) == [1.0, 0.0]
for module in (engine, native):
    path = Path(module.__file__).resolve()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == origins[module.__name__]['sha256']
print(json.dumps({'mode': mode, 'healthy_retry': True,
                  'pressure_exercised': mode not in ('capacity', 'flat_hint', 'outer_hint', 'inner_hint')
                                        and sys.platform == 'linux'}))
"""


def _run_consumer(mode: str, dimensions: tuple[int, int, int] = (1, 1, 1)) -> None:
    """Run one installed consumer and require normal refusal and healthy recovery.

    Parameters
    ----------
    mode
        Capacity or public method whose allocation is exercised.
    dimensions
        Constructor dimensions for the capacity case.
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
            _PROGRAM,
            json.dumps(installed_engine_origins()),
            mode,
            json.dumps(dimensions),
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
        "healthy_retry": True,
        "pressure_exercised": mode not in ("capacity", "flat_hint", "outer_hint", "inner_hint")
        and sys.platform == "linux",
    }


@pytest.mark.parametrize(
    "dimensions",
    [
        (sys.maxsize, 1, 65),
        (1, sys.maxsize, 65),
        (sys.maxsize, sys.maxsize, 1),
        (64, 64, sys.maxsize),
    ],
)
def test_dense_capacity_refusal_and_recovery(dimensions: tuple[int, int, int]) -> None:
    """Refuse unrepresentable dense storage before returning a healthy new layer.

    Parameters
    ----------
    dimensions
        Input, neuron and bitstream dimensions exceeding native storage capacity.
    """
    _run_consumer("capacity", dimensions)


@pytest.mark.parametrize(
    "mode",
    [
        "constructor",
        "get_weights",
        "set_weights",
        "refresh_packed_weights",
        "forward",
        "forward_fast",
        "forward_numpy",
        "forward_numpy_packed",
        "forward_prepacked_list",
        "forward_prepacked_array",
        "forward_prepacked_numpy",
        "forward_batch_numpy",
        "forward_batch_numpy_sample",
    ],
)
def test_dense_allocation_refusal_preserves_state_and_recovers(mode: str) -> None:
    """Refuse bounded Linux allocations and exercise the healthy method elsewhere.

    Parameters
    ----------
    mode
        Public constructor, probability matrix or forward representation to call.
    """
    _run_consumer(mode)


@pytest.mark.parametrize("mode", ["flat_hint", "outer_hint", "inner_hint"])
def test_dense_sequence_hints_do_not_allocate_speculative_storage(mode: str) -> None:
    """Consume finite flat and nested payloads without reserving an oversized hint.

    Parameters
    ----------
    mode
        Flat input, outer weight matrix or inner weight row with a large hint.
    """
    _run_consumer(mode)
