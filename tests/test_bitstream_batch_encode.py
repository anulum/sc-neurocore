# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public seeded bitstream batch encoding contracts

"""Retain distinct scalar and SIMD seeded words, shapes and typed refusal."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt
import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()
import sc_neurocore_engine as engine


class _GoldenCase(TypedDict):
    """Measured public outputs for a length and seed."""

    length: int
    seed: int
    generic: list[list[int]]
    numpy: list[list[int]]


class _GoldenFixture(TypedDict):
    """Probabilities and exact outputs from the recorded engine artifact."""

    schema_version: int
    probabilities: list[float | str]
    cases: list[_GoldenCase]


_FIXTURE = cast(
    _GoldenFixture,
    json.loads((Path(__file__).parent / "fixtures/bitstream_batch_encode_golden.json").read_text()),
)
assert _FIXTURE["schema_version"] == 1
_CASE_IDS = [f"length-{case['length']}-seed-{case['seed']}" for case in _FIXTURE["cases"]]


@pytest.mark.parametrize("case", _FIXTURE["cases"], ids=_CASE_IDS)
@pytest.mark.parametrize("mode", ["generic", "numpy"])
def test_batch_encoding_retains_measured_seeded_words_and_input(
    case: _GoldenCase, mode: str
) -> None:
    """Match previously measured endpoint, nonfinite, SIMD-block and tail traces."""
    probs = np.array([float(value) for value in _FIXTURE["probabilities"]], dtype=np.float64)
    probs.setflags(write=False)
    before = probs.tobytes()
    function = engine.batch_encode if mode == "generic" else engine.batch_encode_numpy
    actual = function(probs, length=case["length"], seed=case["seed"])
    expected = case["generic"] if mode == "generic" else case["numpy"]
    if mode == "generic":
        assert isinstance(actual, list) and actual == expected
    else:
        assert actual.shape == (len(probs), (case["length"] + 63) // 64)
        assert actual.dtype == np.uint64 and actual.flags.c_contiguous
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        function(probs, length=case["length"], seed=case["seed"]), expected
    )
    assert probs.tobytes() == before and not probs.flags.writeable


@pytest.mark.parametrize("length", [0, 65, 1025])
@pytest.mark.parametrize("empty_input", [False, True])
def test_numpy_batch_encoding_owns_output_independently(length: int, empty_input: bool) -> None:
    """Return owning writable matrices, including zero rows and zero columns."""
    probs = np.array([] if empty_input else [0.25, 0.75], dtype=np.float64)
    before = probs.tobytes()
    actual = engine.batch_encode_numpy(probs, length=length, seed=42)
    expected = actual.copy()
    assert actual.shape == (len(probs), (length + 63) // 64)
    assert actual.dtype == np.uint64
    assert actual.flags.owndata and actual.flags.writeable and actual.flags.c_contiguous
    assert not np.shares_memory(actual, probs)
    actual[:] = 0
    assert probs.tobytes() == before
    np.testing.assert_array_equal(
        engine.batch_encode_numpy(probs, length=length, seed=42), expected
    )


@pytest.mark.parametrize("mode", ["generic", "numpy"])
def test_batch_encoding_capacity_refusal_preserves_input_and_retry(mode: str) -> None:
    """Refuse unrepresentable total storage before generating or publishing rows."""
    probs = np.array([0.0, 0.5, 1.0, np.nan], dtype=np.float64)
    probs.setflags(write=False)
    before = probs.tobytes()
    function = engine.batch_encode if mode == "generic" else engine.batch_encode_numpy
    with pytest.raises(MemoryError, match="^encoded bitstreams cannot be allocated$"):
        function(probs, length=2 * sys.maxsize + 1, seed=42)
    assert probs.tobytes() == before
    np.testing.assert_array_equal(function(probs, length=0, seed=42), [[], [], [], []])


@pytest.mark.parametrize("mode", ["generic", "numpy"])
@pytest.mark.parametrize("layout", ["strided", "reversed", "unaligned"])
def test_batch_encoding_retains_layout_refusal_before_sampling(mode: str, layout: str) -> None:
    """Retain readable-buffer admission and caller bytes after actual layout refusal."""
    if layout == "unaligned":
        probs = np.ndarray((2,), dtype=np.float64, buffer=bytearray(17), offset=1)
        probs[:] = [0.25, 0.75]
    else:
        storage = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float64)
        probs = storage[::2] if layout == "strided" else storage[::-1]
    probs.setflags(write=False)
    before = probs.tobytes()
    function = engine.batch_encode if mode == "generic" else engine.batch_encode_numpy
    with pytest.raises(ValueError, match="Cannot read probs"):
        function(probs, length=65, seed=42)
    assert probs.tobytes() == before


@pytest.mark.parametrize("mode", ["generic", "numpy"])
@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.uint8])
def test_batch_encoding_retains_dtype_admission(
    mode: str, dtype: type[np.float32] | type[np.int64] | type[np.uint8]
) -> None:
    """Refuse other dtypes without silently coercing the probability buffer."""
    probs: npt.NDArray[np.float32 | np.int64 | np.uint8] = np.array([0, 1], dtype=dtype)
    before = probs.tobytes()
    function = engine.batch_encode if mode == "generic" else engine.batch_encode_numpy
    with pytest.raises(TypeError):
        function(probs, length=65, seed=42)
    assert probs.tobytes() == before


@pytest.mark.parametrize("mode", ["generic", "numpy"])
@pytest.mark.parametrize("shape", [(), (1, 2)])
def test_batch_encoding_retains_one_dimensional_admission(
    mode: str, shape: tuple[int, ...]
) -> None:
    """Refuse scalar and multidimensional input before output construction."""
    probs = np.zeros(shape, dtype=np.float64)
    before = probs.tobytes()
    function = engine.batch_encode if mode == "generic" else engine.batch_encode_numpy
    with pytest.raises(TypeError):
        function(probs, length=65, seed=42)
    assert probs.tobytes() == before


@pytest.mark.parametrize("configure", [False, True])
def test_cold_parallel_startup_preserves_input_and_retry(configure: bool) -> None:
    """Recover from real Linux worker refusal and retain normal startup elsewhere."""
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
expected = cast(dict[str, dict[str, str]], json.loads(sys.argv[1]))
for name, module in ((engine.__name__, engine), (native.__name__, native)):
    filename = module.__file__
    assert isinstance(filename, str)
    path = Path(filename).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve())
    assert str(path) == expected[name]['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[name]['sha256']
configure = sys.argv[2] == 'True'
probs = np.array([0.0, 1.0], dtype=np.float64)
probs.setflags(write=False)
before = probs.tobytes()
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
        if configure:
            engine.set_num_threads(1)
        else:
            engine.batch_encode_numpy(probs, length=65, seed=42)
    except MemoryError as caught:
        error = caught
    finally:
        resource.setrlimit(resource.RLIMIT_AS, previous)
    assert error is not None
    assert str(error) == 'parallel execution resources cannot be allocated'
    assert probs.tobytes() == before and not probs.flags.writeable
if configure:
    engine.set_num_threads(1)
actual = engine.batch_encode_numpy(probs, length=65, seed=42)
assert actual.tolist() == [[0, 0], [2**64 - 1, 1]]
assert actual.flags.owndata and actual.flags.writeable
assert probs.tobytes() == before and not probs.flags.writeable
try:
    engine.set_num_threads(2)
except ValueError as caught:
    assert str(caught) == 'Cannot set thread pool: The global thread pool has already been initialized.'
else:
    raise AssertionError('an initialized global pool was replaced')
print(json.dumps({'pressure_exercised': pressure, 'configured': configure, 'retry': True}))
"""
    environment = os.environ.copy()
    environment.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            program,
            json.dumps(installed_engine_origins()),
            str(configure),
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
        "pressure_exercised": sys.platform == "linux",
        "configured": configure,
        "retry": True,
    }
