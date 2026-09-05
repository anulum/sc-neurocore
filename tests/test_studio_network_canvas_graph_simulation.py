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
        assert result["schema_version"] == "studio.network-graph-result.v1"
        assert result["graph_summary"]["n_populations"] == 2
        assert result["spec"]["populations"][0]["model"] == "SCLapicqueLIFNeuron"
        assert result["execution"]["backend"]["selected"] == "python"
        assert result["contract"]["kind"] == "network-population-activity"
        # No population declares a drive: an undriven LIF network stays silent.
        assert result["n_spikes"] == 0

    def test_simulate_empty_graph(self):
        result = simulate_graph({"populations": [], "projections": []})
        assert result["success"] is False
        assert "errors" in result

    def test_simulate_three_populations_runs_every_declared_projection(self):
        exc0 = create_population(count=20, neuron_type="excitatory")
        exc1 = create_population(count=20, neuron_type="excitatory")
        inh = create_population(count=10, neuron_type="inhibitory")
        graph = {
            "populations": [exc0, exc1, inh],
            "projections": [
                create_projection(exc0["id"], inh["id"]),
                create_projection(exc1["id"], inh["id"], rule="all_to_all"),
            ],
            "duration": 10.0,
            "dt": 0.1,
        }

        result = simulate_graph(graph)

        assert result["success"] is True
        assert result["graph_summary"]["n_projections"] == 2
        assert [p["count"] for p in result["populations"]] == [20, 20, 10]
        assert result["topology"]["projections"][1]["n_synapses"] == 200

    def test_simulate_rejects_a_sign_conflict_instead_of_flipping_it(self):
        exc = create_population(count=20, neuron_type="excitatory")
        inh = create_population(count=10, neuron_type="inhibitory")
        graph = {
            "populations": [exc, inh],
            "projections": [create_projection(inh["id"], exc["id"], weight=0.5)],
            "duration": 10.0,
            "dt": 0.1,
        }

        result = simulate_graph(graph)

        assert result["success"] is False
        assert any("inhibitory sources need a negative weight" in e for e in result["errors"])
