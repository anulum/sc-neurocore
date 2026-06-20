# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DCLS-max Q8.8 tent kernel — cross-language parity tests

"""Bit-exact parity between every acceleration backend and the Python floor.

The kernel is exact integer Q8.8 arithmetic, so the contract is identical raw
arrays (tolerance zero), not numerical closeness. Each backend test skips with an
explanatory reason when its toolchain artefact is not built in the environment.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from sc_neurocore.scpn.dcls_tent_kernel import (
    DclsBatchResult,
    available_backends,
    dcls_max_forward_batch,
    dcls_max_forward_batch_q88,
)

_BACKENDS = ("rust", "julia", "go", "mojo")


@lru_cache(maxsize=1)
def _availability() -> dict[str, bool]:
    # Probed lazily at first test execution rather than at collection so the
    # juliacall boot does not race torch import ordering during collection.
    return available_backends()


_RESULT_FIELDS = (
    "outputs_q88",
    "accumulators_q16_16",
    "overflow",
    "active_tap_counts",
    "max_gates_q88",
)


def _assert_bit_exact(reference: DclsBatchResult, candidate: DclsBatchResult) -> None:
    for field in _RESULT_FIELDS:
        npt.assert_array_equal(
            getattr(reference, field), getattr(candidate, field), err_msg=field
        )


def _named_workloads() -> dict[str, tuple[Any, Any, Any, Any, int]]:
    workloads: dict[str, tuple[Any, Any, Any, Any, int]] = {}

    # Hand-computed deterministic case.
    workloads["deterministic"] = (
        [1, 1, 1, 0, 1, 0],
        [256, 128, -64, 512, -256, 64],
        [256, 512],
        [512, 768],
        3,
    )

    # All taps silent — every channel contracts to zero.
    n_channels, n_taps = 16, 8
    workloads["all_silent"] = (
        np.zeros(n_channels * n_taps, dtype=np.uint8),
        np.full(n_channels * n_taps, 256, dtype=np.int16),
        np.zeros(n_channels, dtype=np.int16),
        np.full(n_channels, 256, dtype=np.int16),
        n_taps,
    )

    # All taps active at extreme weights — exercises i32 + i16 saturation.
    workloads["saturating"] = (
        np.ones(n_channels * n_taps, dtype=np.uint8),
        np.full(n_channels * n_taps, np.iinfo(np.int16).max, dtype=np.int16),
        np.zeros(n_channels, dtype=np.int16),
        np.full(n_channels, np.iinfo(np.int16).max, dtype=np.int16),
        n_taps,
    )

    # Large randomised batch spanning negative weights and wide tents.
    rng = np.random.default_rng(20260620)
    big_channels, big_taps = 1024, 64
    total = big_channels * big_taps
    workloads["random_large"] = (
        (rng.random(total) < 0.5).astype(np.uint8),
        rng.integers(-32768, 32768, total, dtype=np.int16),
        rng.integers(-256, (big_taps << 8) + 256, big_channels, dtype=np.int16),
        rng.integers(1, (big_taps << 8) + 256, big_channels, dtype=np.int16),
        big_taps,
    )
    return workloads


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("workload_name", list(_named_workloads()))
def test_backend_bit_exact_with_python(backend: str, workload_name: str) -> None:
    if not _availability().get(backend, False):
        pytest.skip(f"{backend} backend not built in this environment")
    spikes, weights, centres, sigmas, n_taps = _named_workloads()[workload_name]
    reference = dcls_max_forward_batch_q88(spikes, weights, centres, sigmas, n_taps)
    candidate = dcls_max_forward_batch(spikes, weights, centres, sigmas, n_taps, backend=backend)
    _assert_bit_exact(reference, candidate)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_backend_rejects_centre_sigma_mismatch(backend: str) -> None:
    if not _availability().get(backend, False):
        pytest.skip(f"{backend} backend not built in this environment")
    with pytest.raises((ValueError, RuntimeError)):
        dcls_max_forward_batch([1, 1], [256, 128], [256, 0], [512], 1, backend=backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_backend_rejects_flat_length_mismatch(backend: str) -> None:
    if not _availability().get(backend, False):
        pytest.skip(f"{backend} backend not built in this environment")
    with pytest.raises((ValueError, RuntimeError)):
        dcls_max_forward_batch([1, 1, 1], [256, 128, -64], [256], [512], 2, backend=backend)


def test_at_least_python_backend_present() -> None:
    assert _availability()["python"] is True
