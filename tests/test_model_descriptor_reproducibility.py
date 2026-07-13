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
import json
import os
import subprocess
import sys

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
import sys

import numpy as np

from sc_neurocore.studio.models import _load_class

model, n_steps, current = sys.argv[1:]
instance = _load_class(model)()
trace, _spikes = instance.simulate(int(n_steps), float(current), backend="python")
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


def _reference_trace(model: str, backend: str) -> npt.NDArray[np.float64]:
    descriptor = _descriptor(model)
    config = json.loads(descriptor.reproducibility.reference_config)
    instance = _load_class(model)()
    trace, _spikes = instance.simulate(
        int(config["n_steps"]), float(config["current"]), backend=backend
    )
    return np.asarray(trace, dtype=float)


def _golden_digest(trace: npt.NDArray[np.float64]) -> str:
    """Return a byte-order-stable SHA-256 digest for a float64 trace."""
    canonical = np.asarray(trace, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _declared_golden_digests(model: str) -> tuple[str, ...]:
    """Return the primary digest and any measured SIMD-equivalent variants."""
    return _descriptor(model).reproducibility.golden_trace_digests


def _reference_trace_without_avx512(model: str) -> npt.NDArray[np.float64]:
    """Re-run a reference trace through NumPy's portable non-AVX-512 path."""
    descriptor = _descriptor(model)
    config = json.loads(descriptor.reproducibility.reference_config)
    n_steps = int(config["n_steps"])
    current = float(config["current"])
    environment = os.environ.copy()
    disabled = set(filter(None, environment.get("NPY_DISABLE_CPU_FEATURES", "").split(",")))
    environment["NPY_DISABLE_CPU_FEATURES"] = ",".join(sorted(disabled | set(_AVX512_FEATURES)))
    completed = subprocess.run(  # nosec B603 - fixed interpreter and audited model catalogue input
        [sys.executable, "-c", _REFERENCE_TRACE_CHILD, model, str(n_steps), repr(current)],
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
    digest = _golden_digest(trace)
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

    assert _golden_digest(native) in declared
    assert _golden_digest(portable) in declared
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
    config = json.loads(descriptor.reproducibility.reference_config)
    checked = 0
    for backend in implemented:
        if backend.name == "python":
            continue
        instance = _load_class(model)()
        try:
            trace, _spikes = instance.simulate(
                int(config["n_steps"]), float(config["current"]), backend=backend.name
            )
        except (RuntimeError, ImportError, OSError):
            continue  # backend not built in this environment — parity tested in CI
        candidate = np.asarray(trace, dtype=float)
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
