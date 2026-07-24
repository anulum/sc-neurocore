# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio network canvas graph simulation

"""Focused suite: TestGraphSimulation from former test_studio_network_canvas.py."""

from __future__ import annotations

from tests.studio_network_canvas_support import *  # noqa: F403


class TestGraphSimulation:
    def test_simulate_valid_graph(self):
        exc = create_population(count=40, neuron_type="excitatory")
        inh = create_population(count=10, neuron_type="inhibitory")
        proj = create_projection(exc["id"], inh["id"])
        graph = {"populations": [exc, inh], "projections": [proj], "duration": 50.0, "dt": 0.1}
        result = simulate_graph(graph)
        assert result["success"] is True
        assert result["n_total"] == 50
        assert "graph_summary" in result

    def test_simulate_empty_graph(self):
        result = simulate_graph({"populations": [], "projections": []})
        assert result["success"] is False
        assert "errors" in result

    def test_simulate_rejects_unsupported_topology(self):
        exc0 = create_population(count=20, neuron_type="excitatory")
        exc1 = create_population(count=20, neuron_type="excitatory")
        inh = create_population(count=10, neuron_type="inhibitory")
        graph = {
            "populations": [exc0, exc1, inh],
            "projections": [create_projection(exc0["id"], inh["id"])],
            "duration": 10.0,
            "dt": 0.1,
        }

        result = simulate_graph(graph)

        assert result["success"] is False
        assert any("exactly one excitatory and one inhibitory" in e for e in result["errors"])
