# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mixed-precision dense MAC — cross-language parity tests

"""Bit-exact parity between every acceleration backend and the Python floor.

The integer mixed-precision dense MAC is exact, so the contract is identical raw
arrays (tolerance zero). Each backend test skips with an explanatory reason when
its toolchain artefact is not built in the environment.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from sc_neurocore.compiler.mixed_dense_kernel import (
    MixedDenseBatchResult,
    available_backends,
    mixed_dense_forward_batch,
    mixed_dense_forward_batch_q88_q1616,
)

_BACKENDS = ("rust", "julia", "go", "mojo")


@lru_cache(maxsize=1)
def _availability() -> dict[str, bool]:
    # Probed lazily at first test execution so the juliacall boot does not race
    # torch import ordering during collection.
    return available_backends()


_RESULT_FIELDS = ("outputs_q1616", "overflow", "underflow")


def _assert_bit_exact(reference: MixedDenseBatchResult, candidate: MixedDenseBatchResult) -> None:
    for field in _RESULT_FIELDS:
        npt.assert_array_equal(getattr(reference, field), getattr(candidate, field), err_msg=field)


def _named_workloads() -> dict[str, tuple[Any, Any, int, int]]:
    workloads: dict[str, tuple[Any, Any, int, int]] = {}

    workloads["deterministic"] = ([256, 128, -64, 512], [512, 1024, 256, 768, 0, 0], 2, 2)

    n_outputs, n_inputs, n_batch = 16, 24, 12
    workloads["all_zero_inputs"] = (
        np.ones(n_outputs * n_inputs, dtype=np.int16),
        np.zeros(n_batch * n_inputs, dtype=np.int32),
        n_outputs,
        n_inputs,
    )

    # Extreme weights and inputs -> i32 saturation on most outputs.
    workloads["saturating"] = (
        np.full(n_outputs * n_inputs, np.iinfo(np.int16).max, dtype=np.int16),
        np.full(n_batch * n_inputs, 2_000_000_000, dtype=np.int32),
        n_outputs,
        n_inputs,
    )

    # Tiny products -> underflow-rich.
    workloads["underflow"] = (
        np.ones(n_outputs * n_inputs, dtype=np.int16),
        np.ones(n_batch * n_inputs, dtype=np.int32),
        n_outputs,
        n_inputs,
    )

    rng = np.random.default_rng(20260621)
    big_outputs, big_inputs, big_batch = 128, 96, 48
    workloads["random_large"] = (
        rng.integers(-32768, 32768, big_outputs * big_inputs, dtype=np.int16),
        rng.integers(-(1 << 21), 1 << 21, big_batch * big_inputs, dtype=np.int32),
        big_outputs,
        big_inputs,
    )
    return workloads


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("workload_name", list(_named_workloads()))
def test_backend_bit_exact_with_python(backend: str, workload_name: str) -> None:
    if not _availability().get(backend, False):
        pytest.skip(f"{backend} backend not built in this environment")
    weights, inputs, n_outputs, n_inputs = _named_workloads()[workload_name]
    reference = mixed_dense_forward_batch_q88_q1616(weights, inputs, n_outputs, n_inputs)
    candidate = mixed_dense_forward_batch(weights, inputs, n_outputs, n_inputs, backend=backend)
    _assert_bit_exact(reference, candidate)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_backend_rejects_weight_length_mismatch(backend: str) -> None:
    if not _availability().get(backend, False):
        pytest.skip(f"{backend} backend not built in this environment")
    with pytest.raises((ValueError, RuntimeError)):
        mixed_dense_forward_batch([1, 1], [1], 1, 1, backend=backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_backend_rejects_input_not_multiple(backend: str) -> None:
    if not _availability().get(backend, False):
        pytest.skip(f"{backend} backend not built in this environment")
    with pytest.raises((ValueError, RuntimeError)):
        mixed_dense_forward_batch([1, 1], [1, 1, 1], 1, 2, backend=backend)


def test_at_least_python_backend_present() -> None:
    assert _availability()["python"] is True
