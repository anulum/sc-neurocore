# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained rational-recovery engine binding

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.neurons.models import sc_clipped_rational_recovery_map
from sc_neurocore.neurons.models.sc_clipped_rational_recovery_map import (
    SCClippedRationalRecoveryMapNeuron,
)

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
ARGS = (0.0, 0.0, 3.0, 0.001, 0.1, 1.0, 1_000_000.0)


def _direct(steps: int, current: float = 0.0) -> tuple[NDArray[np.float64], int, float, float]:
    result: tuple[Any, int, float, float] = extension.py_sc_clipped_rational_recovery_map_simulate(
        *ARGS, steps, current
    )
    trace, events, x_final, y_final = result
    return np.asarray(trace), int(events), float(x_final), float(y_final)


def test_exported_signature_and_bridge_identity() -> None:
    function = extension.py_sc_clipped_rational_recovery_map_simulate
    assert function.__text_signature__ == (
        "(x0, y0, alpha, beta, j, x_threshold, clip_bound, n_steps, current)"
    )
    assert engine.py_sc_clipped_rational_recovery_map_simulate is function
    assert "py_sc_clipped_rational_recovery_map_simulate" in engine.__all__


def test_empty_first_step_and_checked_rejection() -> None:
    empty, events, x_final, y_final = _direct(0)
    assert empty.shape == (0,)
    assert (events, x_final, y_final) == (0, 0.0, 0.0)
    trace, events, x_final, y_final = _direct(1)
    np.testing.assert_array_equal(trace, np.array([0.1]))
    assert (events, x_final, y_final) == (0, 0.1, -0.001)
    with pytest.raises(FloatingPointError, match="invalid"):
        extension.py_sc_clipped_rational_recovery_map_simulate(
            0.0, 0.0, 0.0, 0.001, 0.1, 1.0, 1_000_000.0, 1, 0.0
        )


def test_installed_rust_backend_matches_python() -> None:
    assert sc_clipped_rational_recovery_map._HAS_RUST
    assert (
        sc_clipped_rational_recovery_map._rust_simulate
        is engine.py_sc_clipped_rational_recovery_map_simulate
    )
    rust = SCClippedRationalRecoveryMapNeuron()
    python = SCClippedRationalRecoveryMapNeuron()
    rust_trace, rust_events = rust.simulate(512, backend="rust")
    python_trace, python_events = python.simulate(512, backend="python")
    np.testing.assert_array_equal(rust_trace, python_trace)
    assert (rust_events, rust.x, rust.y) == (python_events, python.x, python.y)
