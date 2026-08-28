# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cazelles source-map engine binding

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.neurons.models import cazelles_map
from sc_neurocore.neurons.models.cazelles_map import CazellesMapNeuron

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
_ARGS = (0.1, 0.0, 0.0, 0.4, 0.6, 0.7, 1.0, 0.0, 1.5, -0.9, 1.4, 1.05, -1.25, 1.5, -1.0)


def _direct(n_steps: int, current: float = 0.0) -> tuple[NDArray[np.float64], int, float]:
    result: tuple[Any, int, float] = extension.py_cazelles_map_simulate(*_ARGS, 2, n_steps, current)
    trace, events, xf = result
    return np.asarray(trace, dtype=np.float64), int(events), float(xf)


def test_exported_signature_and_bridge_identity() -> None:
    function = extension.py_cazelles_map_simulate
    assert function.__text_signature__ == (
        "(x, alpha, x0, x1, x2, x3, x4, a1, a2, a3, a4, b1, b2, b3, b4, exponent, n_steps, current)"
    )
    assert engine.py_cazelles_map_simulate is function
    assert "py_cazelles_map_simulate" in engine.__all__


def test_empty_single_and_checked_failure_contracts() -> None:
    empty, events, xf = _direct(0)
    assert empty.shape == (0,)
    assert empty.dtype == np.float64
    assert (events, xf) == (0, 0.1)
    trace, events, xf = _direct(1)
    np.testing.assert_array_equal(trace, np.array([0.10500000000000001]))
    assert (events, xf) == (0, trace[0])
    with pytest.raises(FloatingPointError, match="candidate left"):
        _direct(1, 2.0)


def test_production_rust_backend_matches_python() -> None:
    assert cazelles_map._HAS_RUST
    assert cazelles_map._rust_simulate is engine.py_cazelles_map_simulate
    rust = CazellesMapNeuron()
    python = CazellesMapNeuron()
    rust_trace, rust_events = rust.simulate(600, backend="rust")
    python_trace, python_events = python.simulate(600, backend="python")
    np.testing.assert_array_equal(rust_trace, python_trace)
    assert (rust_events, rust.x) == (python_events, python.x)
