# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed HDC process memory contracts

"""Exercise HDC allocation refusals under actual Linux process memory limits."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


MEMORY_CONSUMER = """
import json, os, resource, sys
from array import array
from pathlib import Path
from sc_neurocore_engine import BitStreamTensor

operation, mode = sys.argv[1:]
pressure = mode == 'pressure'
original_limit = resource.getrlimit(resource.RLIMIT_AS)
page_size = os.sysconf('SC_PAGE_SIZE')
small = BitStreamTensor.from_packed([1], 1)

def constrain(extra):
    size = int(Path('/proc/self/statm').read_text().split()[0]) * page_size
    soft = size + extra
    assert original_limit[1] == resource.RLIM_INFINITY or soft < original_limit[1]
    resource.setrlimit(resource.RLIMIT_AS, (soft, original_limit[1]))

class RepeatedSequence:
    def __init__(self, value, count, hint, constrain_at_end=False):
        self.value, self.count, self.hint = value, count, hint
        self.calls, self.constrain_at_end = 0, constrain_at_end
    def __len__(self):
        return self.hint
    def __getitem__(self, index):
        self.calls += 1
        if index >= self.count:
            if pressure and self.constrain_at_end:
                constrain(256 * 1024)
            raise IndexError(index)
        return self.value

if operation.startswith('growth_'):
    sequence = RepeatedSequence(1 if operation == 'growth_packed' else small, 1048576, 0)
    if pressure:
        constrain(2 * 1024 * 1024)
elif operation == 'bundle_references':
    sequence = RepeatedSequence(small, 100000, 100000, True)
else:
    word = (1 << 64) - 1 if operation == 'data_integers' else 1
    words = array('Q', [word]) * 262144
    tensor = BitStreamTensor.from_packed(words, len(words) * 64)
    if pressure:
        extra = 256 * 1024
        if operation == 'rotate_packed':
            extra += tensor.length
        elif operation == 'data_python':
            extra += len(words) * 8
        elif operation == 'data_integers':
            extra += len(words) * 16
        constrain(extra)

try:
    if operation == 'growth_packed':
        result = BitStreamTensor.from_packed(sequence, sequence.count * 64)
    elif operation in ('growth_bundle', 'bundle_references'):
        result = BitStreamTensor.bundle(sequence)
    elif operation == 'xor':
        result = tensor.xor(tensor)
    elif operation == 'bundle_one':
        result = BitStreamTensor.bundle([tensor])
    elif operation == 'bundle_three':
        result = BitStreamTensor.bundle([tensor, tensor, tensor])
    elif operation in ('rotate', 'rotate_packed'):
        result = tensor.rotate_right(1)
    elif operation in ('data', 'data_python', 'data_integers'):
        result = tensor.data
    else:
        raise AssertionError(operation)
    outcome = {'kind': 'return'}
except MemoryError as error:
    resource.setrlimit(resource.RLIMIT_AS, original_limit)
    outcome = {'kind': 'MemoryError', 'message': str(error)}
finally:
    resource.setrlimit(resource.RLIMIT_AS, original_limit)

assert BitStreamTensor.from_packed([1], 1).popcount() == 1
outcome['same_process_retry'] = True
if operation.startswith('growth_') or operation == 'bundle_references':
    outcome['sequence_calls'] = sequence.calls
    assert small.data == [1] and small.length == 1
else:
    assert tensor.length == len(words) * 64
    if pressure or operation not in ('rotate', 'rotate_packed'):
        assert tensor.data == words.tolist()
    else:
        assert tensor.data == (array('Q', [2]) * len(words)).tolist()
    outcome['state_verified'] = True
if not pressure:
    if operation == 'xor':
        assert result.length == tensor.length and result.popcount() == 0
    elif operation in ('bundle_one', 'bundle_three'):
        assert result.length == tensor.length and result.data == words.tolist()
    elif operation in ('data', 'data_python', 'data_integers'):
        assert result == words.tolist()
    elif operation in ('rotate', 'rotate_packed'):
        assert result is None
print(json.dumps(outcome), flush=True)
"""


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        ("growth_packed", "cannot allocate HDC input sequence"),
        ("growth_bundle", "cannot allocate HDC input sequence"),
        ("bundle_references", "cannot allocate HDC input sequence"),
        ("xor", "cannot allocate packed XOR output"),
        ("bundle_one", "cannot allocate packed bundle output"),
        ("bundle_three", "cannot allocate packed bundle output"),
        ("rotate", "cannot allocate packed rotation output"),
        ("rotate_packed", "cannot allocate packed rotation output"),
        ("data", "cannot allocate packed data copy"),
        ("data_python", ""),
        ("data_integers", ""),
    ],
)
def test_real_process_memory_refusal_keeps_the_consumer_alive(operation: str, message: str) -> None:
    """Compare a real successful operation with its allocation refusal and retry."""
    for mode in ("control", "pressure"):
        completed = subprocess.run(
            [sys.executable, "-I", "-c", MEMORY_CONSUMER, operation, mode],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
        outcome = json.loads(completed.stdout)
        assert outcome["same_process_retry"] is True
        if mode == "control":
            assert outcome["kind"] == "return"
        else:
            assert outcome["kind"] == "MemoryError"
            assert outcome["message"] == message
            if operation.startswith("growth_"):
                assert 1 < outcome["sequence_calls"] < 1048577
            elif operation == "bundle_references":
                assert outcome["sequence_calls"] == 100001
            else:
                assert outcome["state_verified"] is True
