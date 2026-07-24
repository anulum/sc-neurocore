# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio network canvas validation

"""Focused suite: TestValidation from former test_studio_network_canvas.py."""

from __future__ import annotations

from tests.studio_network_canvas_support import *  # noqa: F403


class TestValidation:
    def _make_graph(self, n_exc=80, n_inh=20):
        exc = create_population(count=n_exc, neuron_type="excitatory")
        inh = create_population(count=n_inh, neuron_type="inhibitory")
        proj = create_projection(exc["id"], inh["id"])
        return {"populations": [exc, inh], "projections": [proj]}

    def test_valid_graph(self):
        errors = validate_graph(self._make_graph())
        assert errors == []

    def test_empty_graph(self):
        errors = validate_graph({"populations": [], "projections": []})
        assert len(errors) > 0
        assert "no populations" in errors[0].lower()

    def test_dangling_projection(self):
        pop = create_population()
        proj = create_projection(pop["id"], "nonexistent")
        errors = validate_graph({"populations": [pop], "projections": [proj]})
        assert any("not found" in e for e in errors)

    def test_zero_weight_warning(self):
        graph = self._make_graph()
        graph["projections"][0]["weight"] = 0
        errors = validate_graph(graph)
        assert any("zero weight" in e for e in errors)

    def test_neuron_count_limit(self):
        graph = self._make_graph(n_exc=1500, n_inh=600)
        errors = validate_graph(graph)
        assert any("2000" in e for e in errors)

    def test_probability_out_of_range(self):
        graph = self._make_graph()
        graph["projections"][0]["probability"] = 1.5
        errors = validate_graph(graph)
        assert any("probability" in e.lower() for e in errors)
