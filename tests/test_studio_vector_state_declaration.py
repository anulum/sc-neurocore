# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - State held as a vector is state and must be declarable

"""State a model holds as a vector must be declarable and recorded.

A compartment vector, a ring-attractor activity profile and a competing-unit
potential array are integration state exactly as a membrane potential is. What
they do not have is a single number to record as their start, and the
descriptor's ``init`` was mandatory — so the generator dropped every one of
them, three models declared no state at all, and their Studio runs were
published with ``complete=false`` for quantities the models track exactly.

Every case here fails on that former behaviour. ``init`` is now absent rather
than fabricated as ``0.0``, because writing a scalar start for a vector states
something about the model that is not true.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.neurons.descriptor_generator import generate_descriptor_payload
from sc_neurocore.neurons.model_descriptor import parse_model_descriptor
from sc_neurocore.studio.models import simulate_model
from sc_neurocore.studio.state_layout import (
    declared_state,
    observe_layout,
    vector_value,
)
from sc_neurocore.studio.models import _load_class

# The three models whose only state is a vector, with the attribute each holds
# it in and the length a default instance carries.
VECTOR_STATE_MODELS: dict[str, tuple[str, int]] = {
    "ContinuousAttractorNeuron": ("u", 16),
    "LeakyCompeteFireNeuron": ("v", 4),
    "RallCableNeuron": ("v", 5),
}

SCALAR_STATE_MODEL = "AdaptiveThresholdIFNeuron"


def _descriptor_payload(state: dict[str, Any]) -> dict[str, Any]:
    """Return a minimal descriptor payload carrying one state table."""
    return {
        "metadata": {
            "schema_version": 2,
            "name": "Probe",
            "class_name": "ProbeNeuron",
            "module": "probe",
        },
        "state": state,
        "parameters": {"tau": {"default": 20.0}},
        "integration": {"dt": 0.1},
    }


class TestVectorValue:
    def test_a_numeric_list_is_a_vector(self) -> None:
        """A model may hold its vector as a list; that is not a statement about state."""
        observed = vector_value([1.0, 2.0, 3.0])
        assert observed is not None
        np.testing.assert_array_equal(observed, np.array([1.0, 2.0, 3.0]))

    def test_a_numeric_tuple_is_a_vector(self) -> None:
        """A tuple of numbers is read as the vector it is."""
        observed = vector_value((1, 2))
        assert observed is not None
        assert observed.shape == (2,)

    def test_an_empty_sequence_is_a_vector_of_length_zero(self) -> None:
        """An empty vector is a vector, not an unreadable value."""
        observed = vector_value([])
        assert observed is not None and observed.shape == (0,)

    def test_an_array_is_still_read_as_before(self) -> None:
        """The array path is unchanged: float, integer and unsigned dtypes read."""
        for array in (
            np.zeros(3, dtype=np.float64),
            np.arange(3, dtype=np.int64),
            np.arange(3, dtype=np.uint8),
        ):
            observed = vector_value(array)
            assert observed is not None and observed.dtype == np.float64

    def test_a_non_numeric_array_is_refused(self) -> None:
        """An array of strings is not a numeric vector."""
        assert vector_value(np.array(["a", "b"])) is None

    def test_a_zero_dimensional_array_is_not_a_vector(self) -> None:
        """A rank-zero array is a scalar, and the scalar path already reads it."""
        assert vector_value(np.float64(1.0).reshape(())) is None

    @pytest.mark.parametrize(
        "value",
        [
            "abc",
            b"abc",
            {"a": 1.0},
            [1.0, "x"],
            [[1.0], [2.0]],
            [True, False],
            None,
            1.0,
        ],
    )
    def test_values_that_are_not_numeric_vectors_are_refused(self, value: object) -> None:
        """A refused value is reported with its reason rather than coerced into a shape."""
        assert vector_value(value) is None


class TestDescriptorInitIsOptional:
    def test_a_state_entry_without_an_init_carries_none(self) -> None:
        """A vector's start is absent, not zero: the descriptor must not invent one."""
        descriptor = parse_model_descriptor(
            _descriptor_payload({"u": {"unit": "1", "meaning": "ring activity"}})
        )
        assert [(s.name, s.init) for s in descriptor.state] == [("u", None)]

    def test_a_state_entry_with_an_init_still_carries_it(self) -> None:
        """The scalar case is unchanged."""
        descriptor = parse_model_descriptor(_descriptor_payload({"v": {"init": -65.0}}))
        assert [(s.name, s.init) for s in descriptor.state] == [("v", -65.0)]

    def test_a_bare_numeric_state_entry_still_carries_it(self) -> None:
        """The shorthand form, a bare number, is unchanged."""
        descriptor = parse_model_descriptor(_descriptor_payload({"v": -70.0}))
        assert [(s.name, s.init) for s in descriptor.state] == [("v", -70.0)]


class TestGeneratorDeclaresVectorState:
    @pytest.mark.parametrize("class_name", sorted(VECTOR_STATE_MODELS))
    def test_the_generator_declares_the_vector_without_an_init(self, class_name: str) -> None:
        """Read from the code, a vector state field is declared and left without a start."""
        attribute, _ = VECTOR_STATE_MODELS[class_name]
        state = generate_descriptor_payload(class_name)["state"]
        assert attribute in state
        assert "init" not in state[attribute]

    def test_a_scalar_state_field_still_carries_its_init(self) -> None:
        """The change adds a case; it does not move the scalar one."""
        state = generate_descriptor_payload(SCALAR_STATE_MODEL)["state"]
        assert state["v"]["init"] == pytest.approx(_load_class(SCALAR_STATE_MODEL)().v)


class TestCommittedDescriptorsDeclareTheirVectors:
    @pytest.mark.parametrize("class_name", sorted(VECTOR_STATE_MODELS))
    def test_the_committed_descriptor_declares_the_vector_with_its_semantics(
        self, class_name: str
    ) -> None:
        """The declaration carries what the generator cannot know: unit and meaning."""
        attribute, _ = VECTOR_STATE_MODELS[class_name]
        source, _, declared = declared_state(class_name)
        assert source == "descriptor"
        by_name = {variable.name: variable for variable in declared}
        assert by_name[attribute].declared_init is None
        assert by_name[attribute].unit != ""
        assert len(by_name[attribute].meaning) > 20

    @pytest.mark.parametrize("class_name", sorted(VECTOR_STATE_MODELS))
    def test_the_vector_is_observed_at_its_real_length(self, class_name: str) -> None:
        """The shape comes from the constructed instance, not from the declaration."""
        attribute, length = VECTOR_STATE_MODELS[class_name]
        source, stem, declared = declared_state(class_name)
        layout = observe_layout(
            _load_class(class_name)(),
            source,
            stem,
            declared,
            n_steps=64,
            element_budget=1 << 20,
        )
        variable = {v.name: v for v in layout.variables}[attribute]
        assert variable.observable
        assert variable.kind == "vector"
        assert variable.shape == (length,)
        assert variable.trace == "per-step"

    def test_a_vector_beyond_the_raw_budget_falls_back_to_snapshots(self) -> None:
        """The budget decision is unchanged for a vector that is now declarable."""
        source, stem, declared = declared_state("RallCableNeuron")
        layout = observe_layout(
            _load_class("RallCableNeuron")(),
            source,
            stem,
            declared,
            n_steps=1000,
            element_budget=10,
        )
        variable = {v.name: v for v in layout.variables}["v"]
        assert variable.observable
        assert variable.trace == "snapshots-only"


class TestRunsAreNowComplete:
    @pytest.mark.parametrize("class_name", sorted(VECTOR_STATE_MODELS))
    def test_a_run_records_the_vector_and_reports_complete_custody(self, class_name: str) -> None:
        """The guarantee that was broken: these runs claimed nothing was declared."""
        attribute, length = VECTOR_STATE_MODELS[class_name]
        result = simulate_model(class_name, use_fast_path=False, duration=2.0)
        layout = result["state_layout"]

        assert layout["source"] == "descriptor"
        assert layout["complete"] is True
        assert layout["incomplete_reasons"] == []
        assert layout["undeclared_mutable"] == []
        assert layout["recorded"] == [attribute]

        trace = np.asarray(result["raw"]["vector_states"][attribute])
        assert trace.shape == (result["n_steps"], length)
        assert len(result["initial_state"][attribute]) == length
        assert len(result["final_state"][attribute]) == length

    def test_the_recorded_trace_is_the_state_the_model_actually_reached(self) -> None:
        """The final snapshot must equal a hand-stepped instance, not a plot sample."""
        result = simulate_model("RallCableNeuron", use_fast_path=False, duration=2.0)
        receipt = result["effective_inputs"]
        assert receipt["protocol"] == "constant"
        reference = _load_class("RallCableNeuron")()
        for _ in range(int(result["n_steps"])):
            reference.step(float(receipt["current"]))
        np.testing.assert_allclose(
            np.asarray(result["final_state"]["v"]), np.asarray(reference.v), rtol=0.0, atol=0.0
        )
