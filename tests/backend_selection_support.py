# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared imports, fixture table and cache controls for backend selection tests

"""Shared imports, a fixture order table, and cache controls for backend selection tests."""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
import platform
from typing import Protocol, cast

import numpy.testing as npt
import pytest

from sc_neurocore.accel import backend_selection as bs
from sc_neurocore.accel.backend_order import FASTEST_FIRST_BACKENDS

_HOST = "Fixture CPU 9000 @ 3.00GHz"
_DCLS = "dcls_max_forward_batch_q88"
_ADC = "adc_to_spike_windows_q"
_MIXED = "mixed_dense_forward_batch_q88_q1616"

#: A fixture table in the packaged form: three unrecorded kernels on one host, and one kernel measured at two levels.
_FIXTURE_TABLE = {
    "schema_version": "sc-neurocore.measured-backend-orders.v1",
    "entries": [
        {
            "kernel": _DCLS,
            "cpu_model": _HOST,
            "isa_level": "unrecorded",
            "order": ["rust", "julia", "mojo", "go"],
        },
        {
            "kernel": _ADC,
            "cpu_model": _HOST,
            "isa_level": "unrecorded",
            "order": ["mojo", "rust", "python"],
        },
        {
            "kernel": _MIXED,
            "cpu_model": _HOST,
            "isa_level": "unrecorded",
            "order": ["julia", "rust", "go"],
        },
        {
            "kernel": "levelled",
            "cpu_model": _HOST,
            "isa_level": "x86-64-v2",
            "order": ["go", "rust"],
        },
        {
            "kernel": "levelled",
            "cpu_model": _HOST,
            "isa_level": "x86-64-v4",
            "order": ["mojo", "go"],
        },
    ],
    "left_out": [],
}

__all__ = [
    "FASTEST_FIRST_BACKENDS",
    "Path",
    "_ADC",
    "_DCLS",
    "_FIXTURE_TABLE",
    "_HOST",
    "_MIXED",
    "_clear_measured_orders_cache",
    "bs",
    "fixture_table",
    "npt",
    "platform",
    "pytest",
]


class _MeasuredOrdersCache(Protocol):
    """Protocol for the cache controls installed by ``functools.cache``."""

    def cache_clear(self) -> None:
        """Clear the cached benchmark-order table."""


def _clear_measured_orders_cache() -> None:
    """Clear the cached table so that a replaced table source is read again."""
    cast(_MeasuredOrdersCache, bs.measured_orders).cache_clear()


@pytest.fixture
def fixture_table(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Replace the packaged table's text with :data:`_FIXTURE_TABLE` for one test."""
    monkeypatch.setattr(bs, "_read_table", lambda: json.dumps(_FIXTURE_TABLE))
    _clear_measured_orders_cache()
    yield
    _clear_measured_orders_cache()
