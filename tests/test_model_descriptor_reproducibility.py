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

import numpy as np
import pytest

from sc_neurocore.neurons.model_catalogue import load_descriptor
from sc_neurocore.neurons.model_descriptor import descriptor_completeness_tier
from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.studio.models import _load_class

_ULP_TOLERANCE = 1e-9


def _tier3_models() -> list[str]:
    return sorted(
        name
        for name in _CLASS_TO_MODULE
        if descriptor_completeness_tier(load_descriptor(name)) >= 3
    )


def _reference_trace(model: str, backend: str) -> np.ndarray:
    descriptor = load_descriptor(model)
    config = json.loads(descriptor.reproducibility.reference_config)
    instance = _load_class(model)()
    trace, _spikes = instance.simulate(
        int(config["n_steps"]), float(config["current"]), backend=backend
    )
    return np.asarray(trace, dtype=float)


def test_at_least_one_model_reaches_tier3() -> None:
    """The polyglot models are engineering-verified, so Tier-3 is non-empty."""

    assert _tier3_models(), "no Tier-3 descriptors — backend/reproducibility regression"


@pytest.mark.parametrize("model", _tier3_models())
def test_golden_trace_is_reproducible(model: str) -> None:
    """Re-running the declared reference config reproduces the golden digest."""

    descriptor = load_descriptor(model)
    trace = _reference_trace(model, "python")
    assert np.all(np.isfinite(trace)), f"{model} reference trace is not finite"
    digest = hashlib.sha256(trace.tobytes()).hexdigest()
    assert digest == descriptor.reproducibility.golden_trace_sha256, (
        f"{model} golden trace drifted from the committed digest"
    )


@pytest.mark.parametrize("model", _tier3_models())
def test_declared_backend_parity_holds(model: str) -> None:
    """Each available declared backend matches Python within its declared parity."""

    descriptor = load_descriptor(model)
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
