# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed dense serialization and copy refusal contracts

"""Preserve dense global identity and live-layer state after persistence refusals."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys

import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()

_PROGRAM = """
import copy
import hashlib
import json
import pickle
from pathlib import Path
import sys

import sc_neurocore_engine as engine
import sc_neurocore_engine.sc_neurocore_engine as native

origins = json.loads(sys.argv[1])
namespace, operation, protocol = sys.argv[2:]
protocol = int(protocol)
for module in (engine, native):
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve())
    assert str(path) == origins[module.__name__]['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == origins[module.__name__]['sha256']
assert engine.DenseLayer is native.DenseLayer
constructor = (engine if namespace == 'facade' else native).DenseLayer
assert constructor.__module__ == 'sc_neurocore_engine.sc_neurocore_engine'
assert constructor.__qualname__ == 'DenseLayer'
engine.set_num_threads(1)
layer = constructor(2, 2, length=65, seed=24301)
control = constructor(2, 2, length=65, seed=24301)
weights = [[1.0, 0.0], [0.0, 1.0]]
for candidate in (layer, control):
    candidate.set_weights(weights)
baseline = control.forward_fast([1.0, 0.0], seed=42)
assert layer.get_weights() == weights
assert layer.forward_fast([1.0, 0.0], seed=42) == baseline
if operation == 'pickle':
    assert pickle.loads(pickle.dumps(constructor, protocol)) is native.DenseLayer
    persist = lambda: pickle.dumps(layer, protocol)
    label = 'DenseLayer' if protocol < 2 else constructor.__module__ + '.DenseLayer'
else:
    assert operation in ('copy', 'deepcopy')
    persist = lambda: getattr(copy, operation)(layer)
    label = constructor.__module__ + '.DenseLayer'
try:
    persist()
except TypeError as error:
    assert str(error) == "cannot pickle '" + label + "' object"
else:
    raise AssertionError('live dense persistence unexpectedly succeeded')
assert layer.get_weights() == weights
assert layer.forward_fast([1.0, 0.0], seed=42) == baseline
layer.set_weights([[0.0, 1.0], [1.0, 0.0]])
assert layer.get_weights() == [[0.0, 1.0], [1.0, 0.0]]
assert layer.forward_fast([1.0, 0.0], seed=42) != baseline
assert control.get_weights() == weights
assert control.forward_fast([1.0, 0.0], seed=42) == baseline
layer.set_weights(weights)
assert layer.forward_fast([1.0, 0.0], seed=42) == baseline
for module in (engine, native):
    digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    assert digest == origins[module.__name__]['sha256']
print(json.dumps({
    'namespace': namespace,
    'operation': operation,
    'protocol': protocol,
    'refusal': 'TypeError',
    'state_preserved': True,
    'independent_instance_preserved': True,
    'healthy_recovery': True,
}))
"""


def _assert_installed_refusal(namespace: str, operation: str, protocol: int) -> None:
    """Run one real installed persistence refusal and require exact recovery evidence.

    Parameters
    ----------
    namespace : str
        Public facade or native namespace used to construct the layer.
    operation : str
        Pickle, shallow copy or deep copy operation under test.
    protocol : int
        Pickle protocol, or -1 when the copy operation chooses its own protocol.
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
            namespace,
            operation,
            str(protocol),
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    observed: object = json.loads(result.stdout.splitlines()[-1])
    assert observed == {
        "namespace": namespace,
        "operation": operation,
        "protocol": protocol,
        "refusal": "TypeError",
        "state_preserved": True,
        "independent_instance_preserved": True,
        "healthy_recovery": True,
    }


@pytest.mark.parametrize("namespace", ("facade", "native"))
@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_dense_pickle_identity_and_instance_refusal(namespace: str, protocol: int) -> None:
    """Round-trip class references and refuse live layers without changing their state.

    Parameters
    ----------
    namespace : str
        Public facade or native alias used to construct the real layer.
    protocol : int
        Every pickle protocol supported by the executing Python interpreter.
    """
    _assert_installed_refusal(namespace, "pickle", protocol)


@pytest.mark.parametrize("namespace", ("facade", "native"))
@pytest.mark.parametrize("operation", ("copy", "deepcopy"))
def test_dense_copy_refusal_preserves_live_instance(namespace: str, operation: str) -> None:
    """Refuse live native copies and retain weights, inference and independent instances.

    Parameters
    ----------
    namespace : str
        Public facade or native alias used to construct the real layer.
    operation : str
        Standard-library shallow or deep copy operation.
    """
    _assert_installed_refusal(namespace, operation, -1)
