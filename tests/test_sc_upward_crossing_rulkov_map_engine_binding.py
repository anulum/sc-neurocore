# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained Rulkov engine-binding contracts

"""Installed-extension contracts for the retained Rulkov identity."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from sc_neurocore.neurons.models.sc_upward_crossing_rulkov_map import (
    SCUpwardCrossingRulkovMapNeuron,
)
from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def test_engine_export_has_stable_signature_and_top_level_identity() -> None:
    """The retained batch symbol must be reachable through the public wheel."""
    function = extension.py_sc_upward_crossing_rulkov_map_simulate

    assert function.__name__ == "py_sc_upward_crossing_rulkov_map_simulate"
    assert function.__text_signature__ == (
        "(x0, y0, alpha, sigma, mu, x_threshold, n_steps, current)"
    )
    assert engine.py_sc_upward_crossing_rulkov_map_simulate is function
    assert "py_sc_upward_crossing_rulkov_map_simulate" in engine.__all__


def test_direct_engine_batch_matches_public_python_model() -> None:
    """The real PyO3 boundary must reproduce the public reference trajectory."""
    result: tuple[Any, int, float, float] = extension.py_sc_upward_crossing_rulkov_map_simulate(
        -1.0, -3.0, 4.0, -1.6, 0.001, 0.25, 2048, 0.5
    )
    trace, events, x_final, y_final = result
    reference = SCUpwardCrossingRulkovMapNeuron(x_threshold=0.25)
    expected_trace, expected_events = reference.simulate(2048, 0.5, backend="python")

    np.testing.assert_array_equal(np.asarray(trace), expected_trace)
    assert events == expected_events
    assert (x_final, y_final) == (reference.x, reference.y)


def test_engine_class_exposes_the_retained_event_sequence() -> None:
    """The top-level wheel class must preserve its count-neutral identity."""
    neuron = engine.SCUpwardCrossingRulkovMapNeuron()

    assert [neuron.step(2.0) for _ in range(3)] == [1, 0, 0]
    assert neuron.get_state() == {"x": -1.0, "y": pytest.approx(-3.0107984)}


def test_engine_rejects_invalid_candidate_without_result() -> None:
    """The PyO3 batch boundary must surface checked Rust arithmetic failure."""
    with pytest.raises(FloatingPointError, match="rejected an invalid candidate"):
        extension.py_sc_upward_crossing_rulkov_map_simulate(
            0.5, 1.0e308, 1.0e308, -1.6, 0.001, 0.0, 2, 0.0
        )
