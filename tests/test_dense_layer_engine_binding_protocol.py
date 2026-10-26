# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed dense input refusal and state contracts

"""Retain measured Dense shape, dtype, layout and caller-conversion refusals."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()

_REFUSALS: tuple[tuple[str, str, str], ...] = (
    ("set_rows", "ValueError", "Expected 2 rows, got 1."),
    ("set_width", "ValueError", "Row 0 has length 1, expected 2."),
    ("set_both", "ValueError", "Expected 2 rows, got 1."),
    ("set_scalar_rows", "TypeError", "'float' object is not an instance of 'Sequence'"),
    ("set_string", "TypeError", "Can't extract `str` to `Vec`"),
    ("set_bad_item", "TypeError", "must be real number, not object"),
    ("set_late_float", "RuntimeError", "caller float conversion refused"),
    ("forward_none", "TypeError", "'None' is not an instance of 'Sequence'"),
    ("forward_generator", "TypeError", "'generator' object is not an instance of 'Sequence'"),
    ("forward_short", "ValueError", "Expected input of length 2, got 1."),
    ("forward_fast_short", "ValueError", "Expected input of length 2, got 1."),
    ("numpy_empty", "ValueError", "Expected input of length 2, got 0."),
    ("numpy_short", "ValueError", "Expected input of length 2, got 1."),
    ("numpy_long", "ValueError", "Expected input of length 2, got 3."),
    ("numpy_f32", "TypeError", "'ndarray' object is not an instance of 'ndarray'"),
    ("numpy_rank2", "TypeError", "'ndarray' object is not an instance of 'ndarray'"),
    (
        "numpy_strided",
        "ValueError",
        "Cannot read input array: The given array is not contiguous or is misaligned.",
    ),
    (
        "numpy_strided_count",
        "ValueError",
        "Cannot read input array: The given array is not contiguous or is misaligned.",
    ),
    ("batch_badfeatures_strided", "ValueError", "Expected 2 input features, got 3."),
    (
        "batch_strided",
        "ValueError",
        "Array not contiguous: The given array is not contiguous or is misaligned.",
    ),
    ("packed_wrong_rows", "ValueError", "Expected 2 packed inputs, got 1."),
    ("packed_wrong_words", "ValueError", "Packed input 0 has 3 words, expected 2."),
    (
        "packed_string",
        "ValueError",
        "packed_inputs must be a 2-D numpy uint64 array or list[list[int]].",
    ),
    (
        "packed_negative",
        "ValueError",
        "packed_inputs must be a 2-D numpy uint64 array or list[list[int]].",
    ),
    ("packed_caller_memory", "MemoryError", "caller integer conversion refused"),
    ("pre_numpy_dtype", "TypeError", "'ndarray' object is not an instance of 'ndarray'"),
    (
        "pre_numpy_strided_count",
        "ValueError",
        "Array not contiguous: The given array is not contiguous or is misaligned.",
    ),
)

_PROGRAM = """
import builtins
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import sc_neurocore_engine as engine
import sc_neurocore_engine.sc_neurocore_engine as native

origins = json.loads(sys.argv[1])
mode, exception_type, message = sys.argv[2:]
for module in (engine, native):
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve())
    assert str(path) == origins[module.__name__]['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == origins[module.__name__]['sha256']
assert engine.DenseLayer is native.DenseLayer
engine.set_num_threads(1)
class RefusedFloat:
    'Exercise an actual caller-owned float-conversion refusal.'
    def __float__(self):
        'Refuse conversion after earlier matrix elements were already consumed.'
        raise RuntimeError('caller float conversion refused')
class RefusedInteger:
    'Preserve a real caller exception during integer protocol conversion.'
    def __index__(self):
        'Refuse integer conversion without simulating allocator exhaustion.'
        raise MemoryError('caller integer conversion refused')
layer = native.DenseLayer(2, 2, length=65, seed=24301)
weights = [[1.0, 0.0], [0.0, 1.0]]
layer.set_weights(weights)
expected = layer.forward_fast([1.0, 0.0], seed=42)
arrays = {
    'numpy_empty': np.ones(0, dtype=np.float64),
    'numpy_short': np.ones(1, dtype=np.float64),
    'numpy_long': np.ones(3, dtype=np.float64),
    'numpy_f32': np.ones(2, dtype=np.float32),
    'numpy_rank2': np.ones((1, 2), dtype=np.float64),
    'numpy_strided': np.ones(4, dtype=np.float64)[::2],
    'numpy_strided_count': np.ones(6, dtype=np.float64)[::2],
    'batch_badfeatures_strided': np.ones((2, 6), dtype=np.float64)[:, ::2],
    'batch_strided': np.ones((2, 4), dtype=np.float64)[:, ::2],
    'packed_wrong_rows': np.ones((1, 2), dtype=np.uint64),
    'packed_wrong_words': np.ones((2, 3), dtype=np.uint64),
    'pre_numpy_dtype': np.ones((2, 2), dtype=np.int64),
    'pre_numpy_strided_count': np.ones((3, 4), dtype=np.uint64)[:, ::2],
}
for array in arrays.values():
    array.setflags(write=False)
before = {key: array.tobytes() for key, array in arrays.items()}
calls = {
    'set_rows': lambda: layer.set_weights([[1.0, 0.0]]),
    'set_width': lambda: layer.set_weights([[1.0], [0.0, 1.0]]),
    'set_both': lambda: layer.set_weights([[1.0]]),
    'set_scalar_rows': lambda: layer.set_weights([1.0, 0.0]),
    'set_string': lambda: layer.set_weights('invalid'),
    'set_bad_item': lambda: layer.set_weights([[object(), 0.0], [0.0, 1.0]]),
    'set_late_float': lambda: layer.set_weights([[0.0, 1.0], [RefusedFloat(), 1.0]]),
    'forward_none': lambda: layer.forward(None, seed=42),
    'forward_generator': lambda: layer.forward((x for x in (1.0, 0.0)), seed=42),
    'forward_short': lambda: layer.forward([1.0], seed=42),
    'forward_fast_short': lambda: layer.forward_fast([1.0], seed=42),
    'numpy_empty': lambda: layer.forward_numpy(arrays['numpy_empty'], seed=42),
    'numpy_short': lambda: layer.forward_numpy(arrays['numpy_short'], seed=42),
    'numpy_long': lambda: layer.forward_numpy(arrays['numpy_long'], seed=42),
    'numpy_f32': lambda: layer.forward_numpy(arrays['numpy_f32'], seed=42),
    'numpy_rank2': lambda: layer.forward_numpy(arrays['numpy_rank2'], seed=42),
    'numpy_strided': lambda: layer.forward_numpy(arrays['numpy_strided'], seed=42),
    'numpy_strided_count': lambda: layer.forward_numpy(arrays['numpy_strided_count'], seed=42),
    'batch_badfeatures_strided': lambda: layer.forward_batch_numpy(arrays['batch_badfeatures_strided'], seed=42),
    'batch_strided': lambda: layer.forward_batch_numpy(arrays['batch_strided'], seed=42),
    'packed_wrong_rows': lambda: layer.forward_prepacked(arrays['packed_wrong_rows']),
    'packed_wrong_words': lambda: layer.forward_prepacked(arrays['packed_wrong_words']),
    'packed_string': lambda: layer.forward_prepacked('invalid'),
    'packed_negative': lambda: layer.forward_prepacked([[-1, 0], [1, 0]]),
    'packed_caller_memory': lambda: layer.forward_prepacked([[1, RefusedInteger()], [1, 0]]),
    'pre_numpy_dtype': lambda: layer.forward_prepacked_numpy(arrays['pre_numpy_dtype']),
    'pre_numpy_strided_count': lambda: layer.forward_prepacked_numpy(arrays['pre_numpy_strided_count']),
}
try:
    calls[mode]()
except Exception as error:
    assert type(error) is getattr(builtins, exception_type)
    assert str(error) == message
else:
    raise AssertionError('invalid input accepted: ' + mode)
assert layer.get_weights() == weights
assert layer.forward_fast([1.0, 0.0], seed=42) == expected
for key, array in arrays.items():
    assert array.tobytes() == before[key] and not array.flags.writeable
replacement = [[0.0, 1.0], [1.0, 0.0]]
layer.set_weights(replacement)
assert layer.get_weights() == replacement
assert layer.forward_fast([1.0, 0.0], seed=42) == [0.0, 1.0]
assert layer.forward([1.0, 0.0], seed=42) == [0.0, 1.0]
np.testing.assert_array_equal(layer.forward_numpy(np.array([1.0, 0.0]), seed=42), [0.0, 1.0])
np.testing.assert_array_equal(layer.forward_batch_numpy(np.eye(2), seed=42), replacement)
packed = np.array([[2**64 - 1, 1], [0, 0]], dtype=np.uint64)
packed.setflags(write=False)
assert layer.forward_prepacked(packed) == [0.0, 1.0]
assert layer.forward_prepacked(packed.tolist()) == [0.0, 1.0]
np.testing.assert_array_equal(layer.forward_prepacked_numpy(packed), [0.0, 1.0])
for module in (engine, native):
    assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == origins[module.__name__]['sha256']
print(json.dumps({'mode': mode, 'refusal_preserved': True, 'state_preserved': True, 'healthy_recovery': True}))
"""


@pytest.mark.parametrize(("mode", "exception_type", "message"), _REFUSALS)
def test_dense_input_refusal_preserves_state_and_recovers(
    mode: str, exception_type: str, message: str
) -> None:
    """Require measured refusal priority, unchanged state and successful replacement.

    Parameters
    ----------
    mode
        Public binding and malformed input representation to exercise.
    exception_type
        Exact exception class observed from the retained installed reference.
    message
        Complete measured diagnostic, including shape-versus-layout priority.
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
            exception_type,
            message,
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
        "refusal_preserved": True,
        "state_preserved": True,
        "healthy_recovery": True,
    }


_ALLOCATOR_PROGRAM = """
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import numpy as np
import sc_neurocore_engine as engine
import sc_neurocore_engine.sc_neurocore_engine as native

origins = json.loads(sys.argv[1])
method, directory, fixture = sys.argv[2:]
for module in (engine, native):
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve())
    assert str(path) == origins[module.__name__]['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == origins[module.__name__]['sha256']
assert engine.DenseLayer is native.DenseLayer
engine.set_num_threads(1)
layer = engine.DenseLayer(2, 2, length=65, seed=24301)
weights = [[0.25, 0.75], [0.75, 0.25]]
layer.set_weights(weights)
arrays = {
    'forward_numpy': np.array([0.25, 0.75], dtype=np.float64),
    'forward_batch_numpy': np.array([[0.25, 0.75], [0.75, 0.25]], dtype=np.float64),
    'forward_prepacked_numpy': np.array(
        [[0x00FF00FF00FF00FF, 1], [0xFF00FF00FF00FF00, 0]], dtype=np.uint64
    ),
}
for array in arrays.values():
    array.setflags(write=False)
before = {name: array.tobytes() for name, array in arrays.items()}
baseline = {name: getattr(layer, name)(array).tolist() for name, array in arrays.items()}
alignment_refusal = sys.platform in ('linux', 'darwin')
if alignment_refusal:
    compiler = shutil.which('cc')
    assert compiler is not None, 'POSIX allocator contracts require a C compiler'
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    assert isinstance(suffix, str)
    binary = Path(directory) / ('dense_numpy_allocator' + suffix)
    command = [
        compiler, '-std=c11', '-shared', '-fPIC', '-O2', '-Wall', '-Wextra', '-Werror',
        '-Wstrict-prototypes', '-Wmissing-prototypes',
        '-I' + sysconfig.get_path('include'), '-I' + np.get_include(),
        fixture, '-o', str(binary),
    ]
    if sys.platform == 'darwin':
        command.extend(['-undefined', 'dynamic_lookup'])
    build = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
    assert build.returncode == 0, (build.returncode, build.stdout, build.stderr)
    assert not build.stderr, build.stderr
    sys.path.insert(0, directory)
    policy = importlib.import_module('dense_numpy_allocator')
    assert Path(policy.__file__).resolve() == binary.resolve()
    original = policy.current()
    previous = policy.install()
    assert previous is original
    try:
        witness = np.zeros((2, 2), dtype=np.float64)
        assert witness.flags.c_contiguous and not witness.flags.aligned
        del witness
        try:
            getattr(layer, method)(arrays[method])
        except ValueError as error:
            assert type(error) is ValueError
            assert str(error) == 'dense output must be contiguous'
        else:
            raise AssertionError('unaligned output unexpectedly reached a Rust slice')
    finally:
        policy.restore(previous)
    assert policy.current() is original
assert layer.get_weights() == weights
for name, array in arrays.items():
    assert array.tobytes() == before[name] and not array.flags.writeable
    assert getattr(layer, name)(array).tolist() == baseline[name]
for module in (engine, native):
    assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == origins[module.__name__]['sha256']
print(json.dumps({
    'method': method,
    'output_alignment_refusal': alignment_refusal,
    'state_preserved': True,
    'healthy_recovery': True,
}))
"""


@pytest.mark.parametrize(
    "method", ("forward_numpy", "forward_batch_numpy", "forward_prepacked_numpy")
)
def test_dense_numpy_output_allocator_refusal_recovers(method: str, tmp_path: Path) -> None:
    """Preserve output alignment refusals and recover after real NumPy policy changes.

    Parameters
    ----------
    method
        Public NumPy inference method whose fresh output is allocated by NumPy.
    tmp_path
        Process-local fixture build directory managed by the test runner.

    Notes
    -----
    Linux and macOS install a real NumPy data allocator and require rejection of
    C-order but unaligned output. Other platforms require the healthy contracts.
    The isolated child restores the exact prior policy before checking recovery.
    """
    environment = os.environ.copy()
    environment.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    environment.pop("RAYON_NUM_THREADS", None)
    fixture = Path(__file__).with_name("fixtures") / "dense_numpy_allocator.c"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _ALLOCATOR_PROGRAM,
            json.dumps(installed_engine_origins()),
            method,
            str(tmp_path),
            str(fixture),
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "method": method,
        "output_alignment_refusal": sys.platform in {"linux", "darwin"},
        "state_preserved": True,
        "healthy_recovery": True,
    }
