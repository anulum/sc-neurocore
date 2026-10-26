# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed dense numerical regression goldens

"""Retain measured nonendpoint Dense outcomes across fallible storage changes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import TypedDict, cast

import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()


class _GoldenCase(TypedDict):
    """Describe one measured constructor and public inference protocol."""

    n_inputs: int
    n_neurons: int
    length: int
    weight_seed: int
    seed: int
    initial_weights_sha256: str
    scalar_counts: list[int]
    fast_counts: list[int]
    prepacked_counts: list[int]
    batch_counts: list[list[int]]


class _GoldenFile(TypedDict):
    """Hold the independently retained installed-reference observations."""

    schema: int
    reference: dict[str, str]
    cases: list[_GoldenCase]


_FIXTURE = Path(__file__).parent / "fixtures" / "dense_layer_forward_golden.json"
_GOLDENS = cast(_GoldenFile, json.loads(_FIXTURE.read_text(encoding="utf-8")))

_PROGRAM = """
import hashlib
import json
from pathlib import Path
import struct
import sys
import numpy as np
import sc_neurocore_engine as engine
import sc_neurocore_engine.sc_neurocore_engine as native

origins, case = json.loads(sys.argv[1]), json.loads(sys.argv[2])
for module in (engine, native):
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve())
    assert str(path) == origins[module.__name__]['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == origins[module.__name__]['sha256']
assert engine.DenseLayer is native.DenseLayer
engine.set_num_threads(1)
n_inputs, n_neurons, length = (case[key] for key in ('n_inputs', 'n_neurons', 'length'))
seed = case['seed']
layer = native.DenseLayer(n_inputs, n_neurons, length=length, seed=case['weight_seed'])
initial = layer.get_weights()
digest = hashlib.sha256(b''.join(struct.pack('<d', value) for row in initial for value in row)).hexdigest()
assert digest == case['initial_weights_sha256']
weights = [[((7 * row + 3 * column) % 17) / 16 for column in range(n_inputs)]
           for row in range(n_neurons)]
layer.set_weights(weights)
values = np.array([((5 * index + 1) % 19) / 20 for index in range(n_inputs)], dtype=np.float64)
maximum = 2**64 - 1
packed = np.array([[(0xA5A5A5A5A5A5A5A5 ^ ((row + 1) << (word % 16))) & maximum
                   for word in range((length + 63) // 64)] for row in range(n_inputs)], dtype=np.uint64)
batch = np.stack([values, 1.0 - values, values[::-1]])
arrays = (values, packed, batch)
for array in arrays:
    array.setflags(write=False)
before = [array.tobytes() for array in arrays]
scalar_expected = np.asarray(case['scalar_counts'], dtype=np.float64) * (1.0 / length)
fast_expected = np.asarray(case['fast_counts'], dtype=np.float64) * (1.0 / length)
packed_expected = np.asarray(case['prepacked_counts'], dtype=np.float64) * (1.0 / length)
batch_expected = np.asarray(case['batch_counts'], dtype=np.float64) * (1.0 / length)
for method, expected in ((layer.forward, scalar_expected), (layer.forward_fast, fast_expected)):
    output = method(values.tolist(), seed=seed)
    assert isinstance(output, list)
    np.testing.assert_array_equal(output, expected)
for payload in (packed.tolist(), packed):
    output = layer.forward_prepacked(payload)
    assert isinstance(output, list)
    np.testing.assert_array_equal(output, packed_expected)
for output, expected in (
    (layer.forward_numpy(values, seed=seed), fast_expected),
    (layer.forward_prepacked_numpy(packed), packed_expected),
    (layer.forward_batch_numpy(batch, seed=seed), batch_expected),
):
    assert output.dtype == np.float64 and output.flags.c_contiguous and output.flags.writeable
    assert all(not np.shares_memory(output, array) for array in arrays)
    np.testing.assert_array_equal(output, expected)
for index, row in enumerate(batch):
    sample_seed = (seed + index * 1_000_000) & maximum
    np.testing.assert_array_equal(layer.forward_fast(row.tolist(), seed=sample_seed), batch_expected[index])
layer.refresh_packed_weights()
assert layer.get_weights() == weights
np.testing.assert_array_equal(layer.forward_fast(values.tolist(), seed=seed), fast_expected)
for array, saved in zip(arrays, before):
    assert array.tobytes() == saved and not array.flags.writeable
for module in (engine, native):
    assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == origins[module.__name__]['sha256']
print(json.dumps({'length': length, 'n_inputs': n_inputs, 'n_neurons': n_neurons,
                  'measured_goldens_retained': True}))
"""


@pytest.mark.parametrize(
    "case",
    _GOLDENS["cases"],
    ids=[
        f"{case['n_inputs']}x{case['n_neurons']}-length-{case['length']}"
        for case in _GOLDENS["cases"]
    ],
)
def test_dense_nonendpoint_goldens_and_wrapping_seeds(case: _GoldenCase) -> None:
    """Retain measured output counts, owned arrays and wrapping sample seeds.

    Parameters
    ----------
    case
        Measured reference outcomes and the dimensions and seeds that produced them.
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
            json.dumps(case),
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "length": case["length"],
        "n_inputs": case["n_inputs"],
        "n_neurons": case["n_neurons"],
        "measured_goldens_retained": True,
    }
