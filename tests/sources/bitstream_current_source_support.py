# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bitstream current-source test support

"""Shared imports and fixtures for bitstream current-source tests."""

import os
import time
from typing import Any

import numpy as np
import pytest

from sc_neurocore.sources.bitstream_current_source import BitstreamCurrentSource

__all__ = ["BitstreamCurrentSource", "_make_source", "_perf_enabled", "np", "pytest", "time"]


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def _make_source(**overrides: Any) -> BitstreamCurrentSource:
    params = dict(
        x_inputs=[0.2, 0.8],
        x_min=0.0,
        x_max=1.0,
        weight_values=[0.5, 0.5],
        w_min=0.0,
        w_max=1.0,
        length=16,
        y_min=0.0,
        y_max=0.1,
        seed=42,
    )
    params.update(overrides)
    return BitstreamCurrentSource(**params)  # type: ignore[arg-type] # Heterogeneous fixture fields
