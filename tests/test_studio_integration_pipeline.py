# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio integration pipeline

"""Focused suite: TestPipeline from former test_studio_integration.py."""

from __future__ import annotations

from typing import Any, cast

from tests.studio_integration_support import *  # noqa: F403


class TestPipeline:
    def _make_graph(self) -> dict[str, object]:
        exc = create_population(count=30, neuron_type="excitatory")
        inh = create_population(count=10, neuron_type="inhibitory")
        proj = create_projection(exc["id"], inh["id"])
        return {"populations": [exc, inh], "projections": [proj], "duration": 30.0}

    def test_pipeline_runs(self) -> None:
        graph = self._make_graph()
        result = run_pipeline(graph)
        assert "steps" in result
        assert "validate" in result["steps"]
        assert "simulate" in result["steps"]

    def test_pipeline_empty_graph(self) -> None:
        result = run_pipeline({"populations": [], "projections": []})
        assert result["success"] is False
        assert result["step"] == "validate"

    def test_pipeline_target(self) -> None:
        graph = self._make_graph()
        result = run_pipeline(graph, target="ecp5")
        assert result.get("target") == "ecp5"

    def test_pipeline_reports_simulation_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        graph = self._make_graph()

        def fail_simulation(_graph: dict[str, Any]) -> dict[str, object]:
            return {"success": False, "errors": ["sim failed"]}

        monkeypatch.setattr(
            "sc_neurocore.studio.network_graph.simulate_graph",
            fail_simulation,
        )

        result = run_pipeline(graph)

        assert result == {
            "success": False,
            "step": "simulate",
            "errors": ["sim failed"],
        }

    def test_the_compile_step_refuses_to_claim_a_hardware_result(self) -> None:
        """The pipeline cannot lower a graph, so it must not report that it did.

        This case replaces one that asserted a bounded ``Compilation failed``
        after monkeypatching the equation compiler to raise. That guarantee —
        a client-facing error that leaks no internals — is preserved below
        against the behaviour that now exists: the step never reaches a
        compiler, because there is no graph lowering to reach it with.
        """
        result = run_pipeline(self._make_graph())

        assert result["success"] is False
        assert result["step"] == "compile"
        assert result["error"] == NO_GRAPH_LOWERING_REASON
        assert result["pipeline"] == "graph → simulate → (no graph lowering)"

    def test_the_refusal_leaks_nothing_about_the_host(self) -> None:
        """The guarantee the replaced case protected, held against the new path."""
        result = run_pipeline(self._make_graph())

        text = str(result)
        assert "Traceback" not in text
        assert "/home/" not in text
        assert "/media/" not in text

    def test_the_simulation_it_did_run_is_still_reported(self) -> None:
        """Refusing the hardware claim must not discard the honest steps."""
        result = run_pipeline(self._make_graph())

        assert result["steps"]["validate"] == {"passed": True}
        assert "n_spikes" in result["steps"]["simulate"]

    def test_a_model_without_recorded_silicon_is_named(self) -> None:
        """A reader must see which declared models carry no lowering at all."""

        def graph_using(model: str) -> dict[str, Any]:
            graph = self._make_graph()
            populations = cast(list[dict[str, Any]], graph["populations"])
            populations[0]["model"] = model
            return graph

        assert run_pipeline(graph_using("AmariNeuralField"))["unsupported_models"] == []
        assert run_pipeline(graph_using("ArcaneNeuron"))["unsupported_models"] == ["ArcaneNeuron"]
