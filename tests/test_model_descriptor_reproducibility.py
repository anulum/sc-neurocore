# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Tier-3 reproducibility and backend-parity gate

"""Engineering-verification gate for Tier-3 descriptors.

A Tier-3 descriptor claims at least two implemented backends and a reproducible
golden trace. This gate makes both claims drift-proof: it re-runs each Tier-3
model's declared reference configuration and checks the golden-trace digest still
matches, and it re-runs every declared backend that is available in the
environment and checks its numeric parity against the Python reference matches
the declared parity. A model whose code drifts from its committed golden trace,
or a backend whose parity silently degrades, fails here.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.model_catalogue import load_descriptor
from sc_neurocore.neurons.model_descriptor import ModelDescriptor, descriptor_completeness_tier
from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.studio.models import _load_class

_ULP_TOLERANCE = 1e-9
_AVX512_FEATURES = (
    "AVX512F",
    "AVX512CD",
    "AVX512_SKX",
    "AVX512_CLX",
    "AVX512_CNL",
    "AVX512_ICL",
)
_REFERENCE_TRACE_CHILD = """
import inspect
import json
import sys

import numpy as np

from sc_neurocore.studio.models import _load_class

model, config_json = sys.argv[1:]
config = json.loads(config_json)
cls = _load_class(model)
constructor = inspect.signature(cls).parameters
kwargs = {key: value for key, value in config.items() if key in constructor}
instance = cls(**kwargs)
result = instance.simulate(config["n_steps"], config["current"], backend="python")
trace = result[0] if isinstance(result, tuple) else result
sys.stdout.buffer.write(np.asarray(trace, dtype="<f8").tobytes())
"""


def _tier3_models() -> list[str]:
    return sorted(
        name for name in _CLASS_TO_MODULE if descriptor_completeness_tier(_descriptor(name)) >= 3
    )


def _descriptor(model: str) -> ModelDescriptor:
    """Load a catalogue descriptor and narrow the registered-model invariant."""
    descriptor = load_descriptor(model)
    assert descriptor is not None, f"{model} is registered without a descriptor"
    return descriptor


def _reference_config(model: str) -> dict[str, object]:
    """Return one descriptor's parsed reference configuration."""
    payload = json.loads(_descriptor(model).reproducibility.reference_config)
    assert isinstance(payload, dict), f"{model} reference_config must be a JSON object"
    return payload


def _construct_instance(model: str, config: dict[str, object]) -> Any:
    """Construct ``model`` with every reference field its constructor declares."""
    cls = _load_class(model)
    constructor = inspect.signature(cls).parameters
    kwargs = {key: value for key, value in config.items() if key in constructor}
    return cls(**kwargs)


def _result_trace(model: str, result: object) -> npt.NDArray[np.float64]:
    """Normalise tuple-returning spiking and array-returning rate simulations."""
    raw_trace = result[0] if isinstance(result, tuple) else result
    trace = np.asarray(raw_trace, dtype=float)
    assert trace.ndim == 1, f"{model} reference simulation returned a non-vector trace"
    return trace


def _truth_table_definition(
    model: str, config: dict[str, object]
) -> tuple[list[str], list[str], list[str], str, list[list[object]]]:
    """Validate and return a declarative truth-table reference configuration."""
    raw_columns = config.get("columns")
    raw_constructor = config.get("constructor_fields")
    raw_simulate = config.get("simulate_fields")
    expected_field = config.get("expected_field")
    raw_rows = config.get("rows")
    assert isinstance(raw_columns, list) and all(isinstance(item, str) for item in raw_columns)
    assert isinstance(raw_constructor, list) and all(
        isinstance(item, str) for item in raw_constructor
    )
    assert isinstance(raw_simulate, list) and all(isinstance(item, str) for item in raw_simulate)
    assert isinstance(expected_field, str)
    assert isinstance(raw_rows, list)
    columns = list(raw_columns)
    constructor_fields = list(raw_constructor)
    simulate_fields = list(raw_simulate)
    rows: list[list[object]] = []
    for raw_row in raw_rows:
        assert isinstance(raw_row, list) and len(raw_row) == len(columns), (
            f"{model} truth-table row does not match its columns"
        )
        rows.append(raw_row)
    assert expected_field in columns
    assert set(constructor_fields + simulate_fields + [expected_field]) <= set(columns)
    return columns, constructor_fields, simulate_fields, expected_field, rows


def _truth_table_trace(
    model: str, backend: str, config: dict[str, object]
) -> npt.NDArray[np.float64]:
    """Execute every declared truth-table row through a model's batch API."""
    columns, constructor_fields, simulate_fields, expected_field, rows = _truth_table_definition(
        model, config
    )
    cls = _load_class(model)
    actual: list[float] = []
    for row in rows:
        values = dict(zip(columns, row, strict=True))
        instance = cls(**{field: values[field] for field in constructor_fields})
        arguments = [[values[field]] for field in simulate_fields]
        trace = _result_trace(model, instance.simulate(*arguments, backend=backend))
        assert trace.shape == (1,), f"{model} truth-table row returned {trace.shape}"
        expected = values[expected_field]
        assert isinstance(expected, (int, float))
        assert float(trace[0]) == float(expected), (
            f"{model}/{backend} disagrees with its declared truth-table row"
        )
        actual.append(float(trace[0]))
    return np.asarray(actual, dtype=float)


def _reference_trace(model: str, backend: str) -> npt.NDArray[np.float64]:
    config = _reference_config(model)
    if config.get("kind") == "truth_table":
        return _truth_table_trace(model, backend, config)
    n_steps = config.get("n_steps")
    assert isinstance(n_steps, int) and not isinstance(n_steps, bool)
    assert "current" in config
    instance = _construct_instance(model, config)
    result = instance.simulate(n_steps, config["current"], backend=backend)
    return _result_trace(model, result)


def _golden_digest(trace: npt.NDArray[np.float64]) -> str:
    """Return a byte-order-stable SHA-256 digest for a float64 trace."""
    canonical = np.asarray(trace, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _truth_table_token(value: object) -> str:
    """Return the canonical text token used by source truth-table evidence."""
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _reference_digest(model: str, trace: npt.NDArray[np.float64]) -> str:
    """Digest a numeric trace or a declared source truth table canonically."""
    config = _reference_config(model)
    if config.get("digest_encoding") == "index_event_state_text_v1":
        n_steps = config.get("n_steps")
        state_field = config.get("state_field")
        assert isinstance(n_steps, int) and not isinstance(n_steps, bool)
        assert isinstance(state_field, str)
        assert "current" in config
        instance = _construct_instance(model, config)
        event_rows: list[str] = []
        replayed: list[float] = []
        for index in range(n_steps):
            event = instance.step(config["current"])
            state = getattr(instance, state_field)
            replayed.append(float(state))
            event_rows.append(f"{index} {int(event)} {_truth_table_token(state)}\n")
        assert np.array_equal(np.asarray(replayed, dtype=float), trace)
        return hashlib.sha256("".join(event_rows).encode()).hexdigest()
    if config.get("kind") != "truth_table":
        return _golden_digest(trace)
    assert config.get("digest_encoding") == "truth_table_text_v1"
    columns, _constructor_fields, _simulate_fields, expected_field, rows = _truth_table_definition(
        model, config
    )
    assert trace.shape == (len(rows),)
    lines = [" ".join(columns) + "\n"]
    for row, output in zip(rows, trace, strict=True):
        values = dict(zip(columns, row, strict=True))
        values[expected_field] = output
        lines.append(" ".join(_truth_table_token(values[column]) for column in columns) + "\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def _declared_golden_digests(model: str) -> tuple[str, ...]:
    """Return the primary digest and any measured SIMD-equivalent variants."""
    return _descriptor(model).reproducibility.golden_trace_digests


def _reference_trace_without_avx512(model: str) -> npt.NDArray[np.float64]:
    """Re-run a reference trace through NumPy's portable non-AVX-512 path."""
    config = _reference_config(model)
    assert config.get("kind") != "truth_table"
    n_steps = config.get("n_steps")
    assert isinstance(n_steps, int) and not isinstance(n_steps, bool)
    environment = os.environ.copy()
    disabled = set(filter(None, environment.get("NPY_DISABLE_CPU_FEATURES", "").split(",")))
    environment["NPY_DISABLE_CPU_FEATURES"] = ",".join(sorted(disabled | set(_AVX512_FEATURES)))
    completed = subprocess.run(  # nosec B603 - fixed interpreter and audited model catalogue input
        [sys.executable, "-c", _REFERENCE_TRACE_CHILD, model, json.dumps(config)],
        check=True,
        capture_output=True,
        env=environment,
    )
    trace = np.frombuffer(completed.stdout, dtype="<f8").copy()
    assert trace.shape == (n_steps,), f"{model} portable reference trace has the wrong length"
    return trace


def test_at_least_one_model_reaches_tier3() -> None:
    """The polyglot models are engineering-verified, so Tier-3 is non-empty."""

    assert _tier3_models(), "no Tier-3 descriptors — backend/reproducibility regression"


@pytest.mark.parametrize("model", _tier3_models())
def test_golden_trace_is_reproducible(model: str) -> None:
    """Re-running the reference config reproduces a measured equivalent digest."""

    trace = _reference_trace(model, "python")
    assert np.all(np.isfinite(trace)), f"{model} reference trace is not finite"
    digest = _reference_digest(model, trace)
    declared = _declared_golden_digests(model)
    assert digest in declared, (
        f"{model} golden trace drifted from every committed platform-equivalent digest"
    )


@pytest.mark.parametrize(
    "model",
    [model for model in _tier3_models() if len(_declared_golden_digests(model)) > 1],
)
def test_declared_simd_golden_variants_remain_ulp_bounded(model: str) -> None:
    """Every allowed SIMD digest must preserve the existing numeric parity bound."""
    native = _reference_trace(model, "python")
    portable = _reference_trace_without_avx512(model)
    declared = _declared_golden_digests(model)

    assert _reference_digest(model, native) in declared
    assert _reference_digest(model, portable) in declared
    max_diff = float(np.max(np.abs(native - portable), initial=0.0))
    assert max_diff < _ULP_TOLERANCE, (
        f"{model} SIMD golden variants exceed the {_ULP_TOLERANCE:.1e} parity bound: {max_diff:.1e}"
    )


@pytest.mark.parametrize("model", _tier3_models())
def test_declared_backend_parity_holds(model: str) -> None:
    """Each available declared backend matches Python within its declared parity."""

    descriptor = _descriptor(model)
    implemented = [b for b in descriptor.backends if b.status == "implemented"]
    assert len(implemented) >= 2, f"{model} declares fewer than two backends"
    reference = _reference_trace(model, "python")
    checked = 0
    for backend in implemented:
        if backend.name == "python":
            continue
        try:
            candidate = _reference_trace(model, backend.name)
        except (RuntimeError, ImportError, OSError):
            continue  # backend not built in this environment — parity tested in CI
        checked += 1
        if backend.parity == "exact":
            assert np.array_equal(candidate, reference), (
                f"{model}/{backend.name} declared exact but diverges from Python"
            )
        elif backend.parity == "ulp-bounded":
            max_diff = float(np.max(np.abs(candidate - reference)))
            assert max_diff < _ULP_TOLERANCE, (
                f"{model}/{backend.name} declared ulp-bounded but max diff {max_diff:.1e}"
            )
        else:  # approximate — just required to run and stay finite
            assert np.all(np.isfinite(candidate)), (
                f"{model}/{backend.name} approximate backend produced non-finite output"
            )
    assert checked >= 1, f"{model} had no runnable non-Python backend to verify"
