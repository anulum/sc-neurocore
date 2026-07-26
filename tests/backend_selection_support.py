# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Backend selection test support

"""Shared imports, constants, and cache controls for backend selection tests."""

from __future__ import annotations

from pathlib import Path
import platform
from typing import Protocol, cast

import numpy.testing as npt
import pytest

from sc_neurocore.accel import backend_selection as bs
from sc_neurocore.accel.backend_order import FASTEST_FIRST_BACKENDS

_RECORDED_CPU = "11th Gen Intel(R) Core(TM) i5-11600K @ 3.90GHz"
_DCLS = "dcls_max_forward_batch_q88"
_ADC = "adc_to_spike_windows_q"
_MIXED = "mixed_dense_forward_batch_q88_q1616"

__all__ = [
    "FASTEST_FIRST_BACKENDS",
    "Path",
    "_ADC",
    "_DCLS",
    "_MIXED",
    "_RECORDED_CPU",
    "_clear_measured_orders_cache",
    "bs",
    "npt",
    "platform",
    "pytest",
]


class _MeasuredOrdersCache(Protocol):
    """Protocol for the cache controls installed by ``functools.cache``."""

    def cache_clear(self) -> None:
        """Clear the cached benchmark-order table."""


def _clear_measured_orders_cache() -> None:
    """Clear benchmark-order cache entries after path monkeypatching."""
    cast(_MeasuredOrdersCache, bs.measured_orders).cache_clear()
