# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed cortical CSR validation and worker recovery

"""Exercise in-place CSR refusal before any row changes and healthy recovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()


@pytest.mark.parametrize("multiple", [False, True])
def test_csr_refusals_preserve_output_before_worker_start(multiple: bool) -> None:
    """Validate late malformed rows and restore real CSR execution after refusal.

    Parameters
    ----------
    multiple
        Single-block or fused multi-block public injection entry point.
    """
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
multiple = sys.argv[2] == 'True'
operation = native.py_parallel_csr_multi_spmv_add if multiple else native.py_parallel_csr_spmv_add
assert engine.py_parallel_csr_spmv_add is native.py_parallel_csr_spmv_add
assert engine.py_parallel_csr_multi_spmv_add is native.py_parallel_csr_multi_spmv_add
rows = 513
indptr = np.arange(rows + 1, dtype=np.int32)
indices = np.arange(rows, dtype=np.int32) % 2
data = (np.arange(rows) % 7 + 1).astype(np.float64)
x = np.array([4., 5.])
data_second = data + 2.0
x_second = np.array([7., 11.])
y = np.full(rows, 0.25)
for array in (indptr, indices, data, x, data_second, x_second):
    array.setflags(write=False)
before = y.tobytes()
inputs_before = [a.tobytes() for a in (indptr, indices, data, x, data_second, x_second)]
components: list[list[npt.NDArray[np.int32] | npt.NDArray[np.float64]]] = [
    [indptr, indptr], [indices, indices], [data, data_second], [x, x_second]]
if multiple:
    arguments = [*components, y]
else:
    arguments = [indptr, indices, data, x, y]
bad_column = indices.copy()
bad_column[-1] = len(x)
negative_column = indices.copy()
negative_column[-1] = -1
negative_offset = indptr.copy()
negative_offset[-1] = -1
descending_offset = indptr.copy()
descending_offset[-1] = descending_offset[-2] - 1
large_offset = indptr.copy()
large_offset[-1] += 1
malformed = [
    (0, indptr[:-1], 'CSR row offsets do not cover the output vector'),
    (0, negative_offset, 'CSR row offsets must be non-negative and nondecreasing'),
    (0, descending_offset, 'CSR row offsets must be non-negative and nondecreasing'),
    (0, large_offset, 'CSR row bounds exceed indices or data storage'),
    (1, indices[:-1], 'CSR row bounds exceed indices or data storage'),
    (2, data[:-1], 'CSR row bounds exceed indices or data storage'),
    (1, bad_column, 'CSR column index out of range'),
    (1, negative_column, 'CSR column index out of range'),
]
engine.set_num_threads(0)
empty_i = np.empty(0, dtype=np.int32)
empty_f = np.empty(0, dtype=np.float64)
assert native.py_parallel_csr_spmv_add(empty_i, empty_i, empty_f, empty_f, empty_f) is None
negative_zero = np.array([-0.0, 2.0])
assert native.py_parallel_csr_multi_spmv_add([], [], [], [], negative_zero) is None
assert negative_zero[0] == 0.0 and not np.signbit(negative_zero[0])
alias = y.copy()
overlapping_storage = np.full(rows + 1, 0.25)
for input_alias, output_alias in ((alias, alias), (overlapping_storage[1:], overlapping_storage[:-1])):
    alias_arguments = list(arguments)
    alias_arguments[3] = [input_alias, x_second] if multiple else input_alias
    alias_arguments[-1] = output_alias
    alias_before = output_alias.tobytes()
    try:
        operation(*alias_arguments)
    except ValueError as caught:
        assert str(caught) == 'y must be writable and must not alias input arrays'
    else:
        raise AssertionError('overlapping input/output borrowing was accepted')
    assert output_alias.tobytes() == alias_before
readonly_y = y.copy()
readonly_y.setflags(write=False)
readonly_arguments = list(arguments)
readonly_arguments[-1] = readonly_y
try:
    operation(*readonly_arguments)
except ValueError as caught:
    assert str(caught) == 'y must be writable and must not alias input arrays'
else:
    raise AssertionError('readonly output borrowing was accepted')
assert readonly_y.tobytes() == before and y.tobytes() == before
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
    for slot, invalid, message in malformed:
        bad_arguments = list(arguments)
        if multiple:
            bad_arguments[slot] = [components[slot][0], invalid]
        else:
            bad_arguments[slot] = invalid
        try:
            operation(*bad_arguments)
        except ValueError as caught:
            assert str(caught) == message
        else:
            raise AssertionError('malformed late CSR row was accepted')
        assert y.tobytes() == before
    if multiple:
        for slot in (1, 2, 3):
            bad_arguments = list(arguments)
            bad_arguments[slot] = [components[slot][0]]
            try:
                operation(*bad_arguments)
            except ValueError as caught:
                assert str(caught) == 'CSR block lists must have matching lengths'
            else:
                raise AssertionError('mismatched CSR block list was accepted')
            assert y.tobytes() == before
    if pressure:
        operation(*arguments)
except MemoryError as caught:
    error = caught
finally:
    if pressure:
        resource.setrlimit(resource.RLIMIT_AS, previous)
if pressure:
    assert error is not None
    assert str(error) == 'parallel execution resources cannot be allocated'
    assert y.tobytes() == before
assert [a.tobytes() for a in (indptr, indices, data, x, data_second, x_second)] == inputs_before
assert operation(*arguments) is None
expected = np.full(rows, 0.25) + data * x[indices]
if multiple:
    expected += data_second * x_second[indices]
np.testing.assert_array_equal(y, expected)
after = y.tobytes()
for slot, invalid, message in malformed:
    bad_arguments = list(arguments)
    if multiple:
        bad_arguments[slot] = [components[slot][0], invalid]
    else:
        bad_arguments[slot] = invalid
    try:
        operation(*bad_arguments)
    except ValueError as caught:
        assert str(caught) == message
    else:
        raise AssertionError('warm malformed CSR row was accepted')
    assert y.tobytes() == after
assert [a.tobytes() for a in (indptr, indices, data, x, data_second, x_second)] == inputs_before
assert all(not a.flags.writeable for a in (indptr, indices, data, x, data_second, x_second))
padded_indices = np.append(indices, np.int32(-1))
padded_data = np.append(data, np.nan)
tail_arguments = list(arguments)
if multiple:
    tail_arguments[1] = [padded_indices, indices]
    tail_arguments[2] = [padded_data, data_second]
else:
    tail_arguments[1] = padded_indices
    tail_arguments[2] = padded_data
assert operation(*tail_arguments) is None
np.testing.assert_array_equal(y, expected + (expected - 0.25))
try:
    engine.set_num_threads(2)
except ValueError as caught:
    assert str(caught) == 'Cannot set thread pool: The global thread pool has already been initialized.'
else:
    raise AssertionError('successful CSR startup did not fix the global pool')
print(json.dumps({'multiple': multiple, 'pressure_exercised': pressure,
                  'output_recovery': True}))
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
            str(multiple),
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
        "multiple": multiple,
        "pressure_exercised": sys.platform == "linux",
        "output_recovery": True,
    }
