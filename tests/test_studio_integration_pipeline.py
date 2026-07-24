# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio integration pipeline

"""Focused suite: TestPipeline from former test_studio_integration.py."""

from __future__ import annotations

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

    def test_pipeline_reports_compile_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        graph = self._make_graph()

        def fail_compile(*_args: object, **_kwargs: object) -> tuple[object, str]:
            raise RuntimeError("compiler failure")

        monkeypatch.setattr(
            "sc_neurocore.compiler.equation_compiler.equation_to_fpga",
            fail_compile,
        )

        result = run_pipeline(graph)

        assert result == {
            "success": False,
            "step": "compile",
            "error": "Compilation failed",
        }
