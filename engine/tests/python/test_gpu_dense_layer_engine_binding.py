# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed GPU dense allocation and retry contracts

"""Exercise a genuinely installed GPU-feature wheel through public Python methods."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()

_ADMISSION = """
import hashlib
import json
from pathlib import Path
import sys
import sc_neurocore_engine as engine
import sc_neurocore_engine.sc_neurocore_engine as native

origins = json.loads(sys.argv[1])
for module in (engine, native):
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve())
    assert str(path) == origins[module.__name__]['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == origins[module.__name__]['sha256']
assert engine.GpuDenseLayer is native.GpuDenseLayer
assert native.GpuDenseLayer.__module__ == 'sc_neurocore_engine.sc_neurocore_engine'
"""

_REFUSAL = (
    _ADMISSION
    + """
case = json.loads(sys.argv[2])
try:
    native.GpuDenseLayer(*case[:4], max_batch=case[4])
except (ValueError, MemoryError, OverflowError) as error:
    assert type(error).__name__ == case[5]
    if case[6] is not None:
        assert str(error) == case[6]
else:
    raise AssertionError('invalid GPU dimensions were admitted')
print(json.dumps({'expected_refusal': case[5]}))
"""
)

_PROTOCOL = (
    _ADMISSION
    + """
import numpy as np
length = int(sys.argv[2])
layer = native.GpuDenseLayer(2, 3, length=length, seed=24301, max_batch=3)
assert native.GpuDenseLayer.is_gpu_available()
name = layer.gpu_name()
assert name and all(word not in name.lower() for word in ('llvmpipe', 'software', 'basic render'))
assert (layer.n_inputs, layer.n_neurons, layer.length) == (2, 3, length)
values = np.array([0.25, 0.75], dtype=np.float64)
values.setflags(write=False)
original = values.tobytes()
reference = layer.forward_fast(values, seed=42)
assert isinstance(reference, list) and len(reference) == 3
assert all(isinstance(value, float) for value in reference)
assert layer.forward_batch(values, 1, seed=42) == reference
strided = np.array([0.25, 0.5, 0.75, 0.5], dtype=np.float64)[::2]
strided.setflags(write=False)
assert layer.forward_fast(strided, seed=42) == reference

class HugeHint:
    def __len__(self):
        raise AssertionError('length hint must not allocate dense storage')
    def __getitem__(self, index):
        return [0.25, 0.75][index]

assert layer.forward_fast(HugeHint(), seed=42) == reference
for payload, samples, message in (
    ([0.25], 1, 'Expected 2 values (1×2), got 1.'),
    ([], 2**64-1, 'Input size overflow when validating n_samples * n_inputs.'),
    ([0.5]*8, 4, 'Batch size 4 exceeds max 3'),
):
    try:
        layer.forward_batch(payload, samples, seed=42)
    except ValueError as error:
        assert str(error) == message
    else:
        raise AssertionError('invalid batch was admitted')
    assert layer.forward_fast(values, seed=42) == reference
assert layer.forward_batch([], 0, seed=42) == []
for seed in (0, 42, 2**64-1):
    single = layer.forward_fast(values, seed=seed)
    assert layer.forward_batch(values, 1, seed=seed) == single
    assert layer.forward_fast(values, seed=seed) == single
owned = layer.forward_fast(values, seed=42)
owned[0] = 999.0
assert layer.forward_fast(values, seed=42) == reference
assert values.tobytes() == original and not values.flags.writeable
print(json.dumps({'length': length, 'gpu': name, 'public_retry_protocol': True}))
"""
)


@pytest.mark.parametrize(
    "symbol", ["GpuDenseLayer", "GpuLifBatch", "GpuKuramoto", "GpuIzhikevichBatch"]
)
def test_gpu_feature_facade_exports_preserve_native_globals(symbol: str) -> None:
    """Exercise optional imports, public star exports and native global pickling.

    Parameters
    ----------
    symbol
        GPU class registered by the compiled feature profile.
    """
    program = (
        _ADMISSION
        + """
import pickle
symbol = sys.argv[2]
value = getattr(engine, symbol)
assert value is getattr(native, symbol)
assert value.__module__ == 'sc_neurocore_engine.sc_neurocore_engine'
namespace = {}
exec('from sc_neurocore_engine import *', namespace)
assert namespace[symbol] is value
assert pickle.loads(pickle.dumps(value)) is value
print(json.dumps({'symbol': symbol, 'native_identity_and_global_pickle': True}))
"""
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, json.dumps(installed_engine_origins()), symbol],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "symbol": symbol,
        "native_identity_and_global_pickle": True,
    }


@pytest.mark.parametrize(
    "case",
    [
        (0, 0, 0, 24301, 0, "ValueError", "bitstream length must be > 0"),
        (0, 0, 1, 24301, 0, "ValueError", "n_inputs must be > 0"),
        (1, 0, 1, 24301, 0, "ValueError", "n_neurons must be > 0"),
        (1, 1, 1, 24301, 0, "ValueError", "max_batch must be > 0"),
        (2**64 - 1, 2, 64, 42, 1, "MemoryError", "dense storage cannot be allocated"),
        (1, 2**64 - 1, 64, 42, 1, "MemoryError", "dense storage cannot be allocated"),
        (1, 1, 64, 42, 2**64 - 1, "MemoryError", "dense storage cannot be allocated"),
        (64, 2, 2**64 - 1, 42, 2, "MemoryError", "dense storage cannot be allocated"),
        (1, 1, 2**32, 42, 1, "ValueError", "GPU dense dimensions exceed u32 shader indexing"),
        (-1, 1, 64, 42, 1, "OverflowError", None),
        (1, 1, -1, 42, 1, "OverflowError", None),
        (1, 1, 64, 42, -1, "OverflowError", None),
    ],
)
def test_gpu_dense_constructor_rejections(
    case: tuple[int, int, int, int, int, str, str | None],
) -> None:
    """Propagate authored dimension, conversion and capacity failures from the wheel.

    Parameters
    ----------
    case
        Constructor arguments and the exact expected public exception contract.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _REFUSAL,
            json.dumps(installed_engine_origins()),
            json.dumps(case),
        ],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert json.loads(result.stdout.splitlines()[-1]) == {"expected_refusal": case[5]}


@pytest.mark.parametrize("length", [1, 63, 65, 129])
def test_gpu_dense_public_sequences_and_error_atomicity(length: int) -> None:
    """Exercise real GPU buffers, owned outputs and healthy retries at word boundaries.

    Parameters
    ----------
    length
        Bitstream length spanning short streams and partially filled final words.
    """
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _PROTOCOL,
            json.dumps(installed_engine_origins()),
            str(length),
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["length"] == length and report["public_retry_protocol"] is True


def test_gpu_dense_unavailable_adapter_selection() -> None:
    """Retain the real adapter-selection refusal through the installed GPU binding."""
    environment = os.environ.copy()
    environment.update(SC_GPU_ADAPTER="SC-NeuroCore unavailable adapter selection")
    program = (
        _ADMISSION
        + """
assert not native.GpuDenseLayer.is_gpu_available()
try:
    native.GpuDenseLayer(2, 3, length=65, seed=24301, max_batch=1)
except RuntimeError as error:
    assert str(error) == 'No GPU available. Check Vulkan/Metal drivers.'
else:
    raise AssertionError('an unmatched actual adapter selector was admitted')
print(json.dumps({'unavailable_adapter_selection': True}))
"""
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, json.dumps(installed_engine_origins())],
        capture_output=True,
        text=True,
        env=environment,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert json.loads(result.stdout.splitlines()[-1]) == {"unavailable_adapter_selection": True}
