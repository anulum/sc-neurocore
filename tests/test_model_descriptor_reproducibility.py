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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.model_catalogue import load_descriptor
from sc_neurocore.neurons.model_descriptor import ModelDescriptor, descriptor_completeness_tier
from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.studio.models import _load_class

_ULP_TOLERANCE = 1e-9
_P = ParamSpec("_P")
_R = TypeVar("_R")
_AVX512_FEATURES = (
    "AVX512F",
    "AVX512CD",
    "AVX512_SKX",
    "AVX512_CLX",
    "AVX512_CNL",
    "AVX512_ICL",
)
_DEDICATED_REPRODUCIBILITY_MODELS = {
    "AiharaMapNeuron",
    "BendaHerzNeuron",
    "CompteWMNeuron",
    "EnergyLIFNeuron",
    "GLMNeuron",
    "HillTononiNeuron",
    "IhNeuron",
    "MainenSejnowskiNeuron",
    "MATNeuron",
    "McKeanNeuron",
    "NMDANeuron",
    "NonResettingLIFNeuron",
    "PersistentNaNeuron",
    "SigmaDeltaNeuron",
    "SKNeuron",
    "TTypeCaNeuron",
}


def _parametrize(
    argnames: str,
    argvalues: object,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Return a typed view of pytest's externally untyped parametriser."""
    marker = cast(
        "Callable[[str, object], Callable[[Callable[_P, _R]], Callable[_P, _R]]]",
        pytest.mark.parametrize,
    )
    return marker(argnames, argvalues)


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
if config.get("kind") == "sampled_batch_v1":
    arguments = []
    for field in config["simulate_fields"]:
        spec = config["inputs"][field]
        index = np.arange(config["n_steps"] * spec.get("length_multiplier", 1), dtype=float)
        angle = spec.get("angular_frequency", 0.0) * index + spec.get("phase", 0.0)
        function = spec.get("function", "constant")
        wave = np.sin(angle) if function == "sin" else np.cos(angle) if function == "cos" else np.ones_like(index)
        arguments.append(spec.get("offset", 0.0) + spec.get("amplitude", 0.0) * wave)
    result = instance.simulate(*arguments, backend="python")
    trace = np.column_stack([np.asarray(result[field], dtype=float) for field in config["trace_fields"]]).reshape(-1)
else:
    result = instance.simulate(config["n_steps"], config["current"], backend="python")
    trace = result[0] if isinstance(result, tuple) else result
sys.stdout.buffer.write(np.asarray(trace, dtype="<f8").tobytes())
"""


def _all_tier3_models() -> list[str]:
    return sorted(
        name for name in _CLASS_TO_MODULE if descriptor_completeness_tier(_descriptor(name)) >= 3
    )


def _supports_generic_reproducibility(model: str) -> bool:
    """Return whether the descriptor fits this generic batch replay harness."""
    raw_config = _descriptor(model).reproducibility.reference_config.strip()
    if not raw_config.startswith("{"):
        return False
    payload = json.loads(raw_config)
    if not isinstance(payload, dict) or not callable(getattr(_load_class(model), "simulate", None)):
        return False
    if payload.get("kind") == "sampled_batch_v1":
        return all(
            key in payload for key in ("n_steps", "simulate_fields", "inputs", "trace_fields")
        )
    if payload.get("kind") == "truth_table":
        return all(
            key in payload
            for key in (
                "columns",
                "constructor_fields",
                "simulate_fields",
                "expected_field",
                "rows",
            )
        )
    return "n_steps" in payload and "current" in payload


def _tier3_models() -> list[str]:
    """Return Tier-3 models supported by the generic descriptor replay route."""
    return [model for model in _all_tier3_models() if _supports_generic_reproducibility(model)]


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


def _result_trace(
    model: str,
    result: object,
    trace_fields: Sequence[str] = (),
) -> npt.NDArray[np.float64]:
    """Normalise scalar or named multi-observable simulation results."""
    if trace_fields:
        assert isinstance(result, Mapping), (
            f"{model} sampled batch must return a mapping for named trace fields"
        )
        columns: list[npt.NDArray[np.float64]] = []
        expected_shape: tuple[int, ...] | None = None
        for field in trace_fields:
            assert field in result, f"{model} sampled batch omitted trace field {field!r}"
            column = np.asarray(result[field], dtype=float)
            assert column.ndim == 1, f"{model}.{field} is not a vector"
            if expected_shape is None:
                expected_shape = column.shape
            assert column.shape == expected_shape, f"{model} sampled trace shapes disagree"
            columns.append(column)
        assert columns, f"{model} sampled batch declared no trace fields"
        return np.asarray(np.column_stack(columns).reshape(-1), dtype=float)
    raw_trace = result[0] if isinstance(result, tuple) else result
    trace = np.asarray(raw_trace, dtype=float)
    assert trace.ndim == 1, f"{model} reference simulation returned a non-vector trace"
    return trace


def _numeric_spec_value(
    model: str,
    field: str,
    spec: Mapping[str, object],
    key: str,
    default: float,
) -> float:
    """Return one finite numeric sampled-input setting."""
    value = spec.get(key, default)
    assert isinstance(value, (int, float)) and not isinstance(value, bool), (
        f"{model}.{field}.{key} must be numeric"
    )
    converted = float(value)
    assert np.isfinite(converted), f"{model}.{field}.{key} must be finite"
    return converted


def _sampled_input(
    model: str,
    field: str,
    raw_spec: object,
    n_steps: int,
) -> npt.NDArray[np.float64]:
    """Build one safe declarative sinusoid, cosinusoid, or constant input."""
    assert isinstance(raw_spec, Mapping), f"{model}.{field} sampled input must be an object"
    multiplier = raw_spec.get("length_multiplier", 1)
    assert isinstance(multiplier, int) and not isinstance(multiplier, bool) and multiplier > 0, (
        f"{model}.{field}.length_multiplier must be a positive integer"
    )
    function = raw_spec.get("function", "constant")
    assert function in {"constant", "sin", "cos"}, (
        f"{model}.{field}.function must be constant, sin, or cos"
    )
    offset = _numeric_spec_value(model, field, raw_spec, "offset", 0.0)
    amplitude = _numeric_spec_value(model, field, raw_spec, "amplitude", 0.0)
    angular_frequency = _numeric_spec_value(model, field, raw_spec, "angular_frequency", 0.0)
    phase = _numeric_spec_value(model, field, raw_spec, "phase", 0.0)
    index: npt.NDArray[np.float64] = np.arange(
        n_steps * multiplier,
        dtype=np.float64,
    )
    angle = angular_frequency * index + phase
    if function == "sin":
        wave = np.sin(angle)
    elif function == "cos":
        wave = np.cos(angle)
    else:
        wave = np.ones_like(index)
    return np.asarray(offset + amplitude * wave, dtype=float)


def _sampled_batch_trace(
    model: str,
    backend: str,
    config: dict[str, object],
) -> npt.NDArray[np.float64]:
    """Execute a declarative multi-input batch and interleave named traces."""
    n_steps = config.get("n_steps")
    raw_fields = config.get("simulate_fields")
    raw_inputs = config.get("inputs")
    raw_trace_fields = config.get("trace_fields")
    assert isinstance(n_steps, int) and not isinstance(n_steps, bool) and n_steps >= 0
    assert isinstance(raw_fields, list) and all(isinstance(field, str) for field in raw_fields)
    assert isinstance(raw_inputs, Mapping)
    assert isinstance(raw_trace_fields, list) and all(
        isinstance(field, str) for field in raw_trace_fields
    )
    fields = list(raw_fields)
    trace_fields = list(raw_trace_fields)
    assert fields, f"{model} sampled batch declared no simulate fields"
    assert len(set(fields)) == len(fields), f"{model} sampled batch fields must be unique"
    assert len(set(trace_fields)) == len(trace_fields), (
        f"{model} sampled trace fields must be unique"
    )
    arguments = [_sampled_input(model, field, raw_inputs.get(field), n_steps) for field in fields]
    instance = _construct_instance(model, config)
    result = instance.simulate(*arguments, backend=backend)
    trace = _result_trace(model, result, trace_fields)
    assert trace.shape == (n_steps * len(trace_fields),), (
        f"{model} sampled batch returned {trace.shape}"
    )
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
    if config.get("kind") == "sampled_batch_v1":
        return _sampled_batch_trace(model, backend, config)
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
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join((source_root, inherited_pythonpath))
        if inherited_pythonpath
        else source_root
    )
    disabled = set(filter(None, environment.get("NPY_DISABLE_CPU_FEATURES", "").split(",")))
    environment["NPY_DISABLE_CPU_FEATURES"] = ",".join(sorted(disabled | set(_AVX512_FEATURES)))
    completed = subprocess.run(  # nosec B603 - fixed interpreter and audited model catalogue input
        [sys.executable, "-c", _REFERENCE_TRACE_CHILD, model, json.dumps(config)],
        check=True,
        capture_output=True,
        env=environment,
    )
    trace = np.frombuffer(completed.stdout, dtype="<f8").copy()
    trace_fields = config.get("trace_fields", ())
    if config.get("kind") == "sampled_batch_v1":
        assert isinstance(trace_fields, list) and all(
            isinstance(field, str) for field in trace_fields
        )
        expected_length = n_steps * len(trace_fields)
    else:
        expected_length = n_steps
    assert trace.shape == (expected_length,), (
        f"{model} portable reference trace has the wrong length"
    )
    return trace


def test_at_least_one_model_reaches_tier3() -> None:
    """The polyglot models are engineering-verified, so Tier-3 is non-empty."""

    assert _tier3_models(), "no Tier-3 descriptors — backend/reproducibility regression"


def test_non_generic_tier3_models_have_dedicated_reproducibility_routes() -> None:
    """Every Tier-3 model outside the generic batch shape is explicitly routed."""

    dedicated = set(_all_tier3_models()) - set(_tier3_models())
    assert dedicated == _DEDICATED_REPRODUCIBILITY_MODELS


@_parametrize("model", _tier3_models())
def test_golden_trace_is_reproducible(model: str) -> None:
    """Re-running the reference config reproduces a measured equivalent digest."""

    trace = _reference_trace(model, "python")
    assert np.all(np.isfinite(trace)), f"{model} reference trace is not finite"
    digest = _reference_digest(model, trace)
    declared = _declared_golden_digests(model)
    assert digest in declared, (
        f"{model} golden trace drifted from every committed platform-equivalent digest"
    )


@_parametrize(
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


@_parametrize("model", _tier3_models())
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
