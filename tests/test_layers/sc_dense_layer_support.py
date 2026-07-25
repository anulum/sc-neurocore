# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCDenseLayer test support

"""Shared imports and fixtures for SCDenseLayer tests."""

import os
import time
from typing import Any

import numpy as np
import pytest

from sc_neurocore.layers.sc_dense_layer import SCDenseLayer

__all__ = ["SCDenseLayer", "_make_layer", "_perf_enabled", "np", "pytest", "time"]


def _make_layer(**overrides: Any) -> SCDenseLayer:
    params = dict(
        n_neurons=2,
        x_inputs=[0.2, 0.4],
        weight_values=[0.5, 0.5],
        x_min=0.0,
        x_max=1.0,
        w_min=0.0,
        w_max=1.0,
        length=16,
        dt_ms=1.0,
        neuron_params={"noise_std": 0.0, "tau_mem": 1e9},
        base_seed=123,
    )
    params.update(overrides)
    return SCDenseLayer(**params)  # type: ignore[arg-type] # Heterogeneous fixture fields


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"
