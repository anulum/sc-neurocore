# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The Amari field's site count is a parameter

"""The number of ring sites is configuration, not a per-step state variable.

``AmariNeuralField`` declared ``n`` — the number of uniformly spaced sites on
the periodic ring — in its descriptor's state table with ``init = 64.0``. The
site count is fixed at construction and read by every step without ever being
written, so a run recorded a constant as a per-step trace, and the runtime
state census reported it as state the Rust batch lane had *dropped*: a
conformance gap that existed only because the descriptor asked a lane to carry
a number that never moves. The model's own documentation page has always listed
``n`` under Parameters.

Nothing in the curated schema said it was state. The generator recognises the
name because ``n`` is the Hodgkin-Huxley potassium activation gate, and a field
whose name is recognised and whose dynamics never assign it still falls through
to the state table. The fix is a decision about this one identity, so the cases
here also hold the other half: where ``n`` really is a gating variable it is
still state, and the classifier was not widened or narrowed to reach either
answer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.neurons.descriptor_generator import generate_descriptor_payload
from sc_neurocore.neurons.model_catalogue import load_descriptor
from sc_neurocore.neurons.model_descriptor import descriptor_completeness_tier
from sc_neurocore.neurons.models.amari_field import AmariNeuralField
from tools.runtime_state_conformance import build_matrix

#: Models where ``n`` is the potassium activation gate and must stay state.
GATING_MODELS = ("HodgkinHuxleyNeuron", "ConnorStevensNeuron")


@pytest.fixture(scope="module")
def descriptor() -> Any:
    """The committed Amari descriptor."""
    committed = load_descriptor("AmariNeuralField")
    assert committed is not None
    return committed


class TestTheSiteCountIsConfiguration:
    def test_the_state_table_holds_only_the_field(self, descriptor: Any) -> None:
        """``u`` is what evolves; the site count is not part of the trace."""
        assert [variable.name for variable in descriptor.state] == ["u"]

    def test_the_site_count_is_a_parameter(self, descriptor: Any) -> None:
        """It has to be somewhere — as configuration, with its default intact."""
        sites = {parameter.name: parameter for parameter in descriptor.parameters}["n"]
        assert sites.default == 64.0

    def test_the_generator_classifies_it_without_the_committed_file(self) -> None:
        """A hand-edited descriptor would pass the cases above; this one needs the code."""
        payload = generate_descriptor_payload("AmariNeuralField")
        assert "n" in payload["parameters"]
        assert "n" not in payload["state"]

    def test_the_site_count_never_moves_during_a_run(self) -> None:
        """The claim being made: a hundred steps change the field and not the count."""
        neuron = AmariNeuralField(n=8)
        before = neuron.n
        for _ in range(100):
            neuron.step(1.0)
        assert neuron.n == before
        assert np.any(neuron.u != 0.0)


class TestTheCurationIsComplete:
    def test_it_carries_unit_range_and_meaning(self, descriptor: Any) -> None:
        """A parameter without these drops the model out of the curated tier."""
        sites = {parameter.name: parameter for parameter in descriptor.parameters}["n"]
        assert sites.unit == "sites"
        assert sites.meaning
        assert sites.value_range is not None
        assert sites.is_curated

    def test_the_model_keeps_its_completeness_tier(self, descriptor: Any) -> None:
        """Moving a variable between tables must not cost the descriptor a tier."""
        assert all(parameter.is_curated for parameter in descriptor.parameters)
        assert descriptor_completeness_tier(descriptor) == 3

    def test_the_declared_floor_is_the_site_count_guard_s_own_bound(self, descriptor: Any) -> None:
        """A per-parameter range states what the count alone has to satisfy.

        Two and three sites are refused at the default kernel widths, but by
        the joint excitation/inhibition constraint across ``dx`` and the two
        widths, not by the count. So the floor a per-parameter range can
        declare is the guard's own, and a case demanding a constructible model
        at ``n = 2`` would be asserting something about the other parameters.
        """
        sites = {parameter.name: parameter for parameter in descriptor.parameters}["n"]
        assert sites.value_range is not None
        low, high = sites.value_range
        assert low <= sites.default <= high
        with pytest.raises(ValueError, match="n must be an integer greater than or equal to two"):
            AmariNeuralField(n=int(low) - 1)
        with pytest.raises(ValueError, match="kernel must be"):
            AmariNeuralField(n=int(low))

    def test_the_declared_ceiling_is_the_interaction_matrix_budget(self, descriptor: Any) -> None:
        """The model materialises an n-by-n matrix, so the ceiling is memory.

        Nothing refuses a larger count at runtime; the bound is a statement of
        where the dense interaction matrix stops being allocatable on an
        ordinary machine, and it is exact — one site more crosses the budget.
        """
        budget_bytes = 512 * 1024 * 1024
        sites = {parameter.name: parameter for parameter in descriptor.parameters}["n"]
        assert sites.value_range is not None
        _, high = sites.value_range
        assert int(high) ** 2 * 8 == budget_bytes
        assert (int(high) + 1) ** 2 * 8 > budget_bytes
        assert AmariNeuralField(n=32)._interaction.nbytes == 32**2 * 8


class TestTheRecognisedNameStillMeansStateElsewhere:
    @pytest.mark.parametrize("class_name", GATING_MODELS)
    def test_a_gating_variable_named_n_is_still_state(self, class_name: str) -> None:
        """Fixing one identity by weakening the classifier would break these."""
        committed = load_descriptor(class_name)
        assert committed is not None
        assert "n" in [variable.name for variable in committed.state]


class TestTheCensusNoLongerReportsIt:
    def test_the_lane_is_not_charged_with_dropping_a_constant(self) -> None:
        """The conformance gap the misclassification manufactured is gone."""
        rows = build_matrix()["rows"]
        assert isinstance(rows, list)
        row = next(entry for entry in rows if entry["model"] == "AmariNeuralField")
        assert row["declared"] == ["u"]
        for lane in row["lanes"].values():
            assert "n" not in lane["dropped"]
