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
        exc = create_population(count=30, neuron_type="excitatory", model="AdExNeuron")
        inh = create_population(count=10, neuron_type="inhibitory", model="AdExNeuron")
        proj = create_projection(exc["id"], inh["id"])
        return {"populations": [exc, inh], "projections": [proj], "duration": 30.0}

    def _supported_graph(self) -> dict[str, Any]:
        """Two small integrator populations with exactly representable values."""
        src = create_population(
            count=2,
            model="PerfectIntegratorNeuron",
            params={"c_m": 1.0, "v_threshold": 1.0},
            drive={"kind": "constant", "current": 0.375},
        )
        dst = create_population(
            count=2, model="PerfectIntegratorNeuron", params={"c_m": 1.0, "v_threshold": 1.0}
        )
        proj = create_projection(src["id"], dst["id"], weight=0.5, delay=2.0)
        return {"populations": [src, dst], "projections": [proj], "duration": 12.0, "dt": 1.0}

    def test_a_supported_network_reaches_synthesis_with_its_trace(self) -> None:
        result = run_pipeline(self._supported_graph())

        assert result["pipeline"] == PIPELINE_ROUTE
        assert list(result["steps"]) == [
            "validate",
            "simulate",
            "lower",
            "cosimulate",
            "synthesise",
        ]
        cosim = result["steps"]["cosimulate"]
        assert cosim["rtl_matches_bit_true_model"] is True
        assert cosim["studio_agreement"] == {"identical": True, "first_divergent_step": None}
        assert cosim["steps"] == 12
        assert result["trace"]["input_sha256"] == result["steps"]["lower"]["input_sha256"]
        assert set(result["trace"]) == {
            "input_sha256",
            "rtl_sha256",
            "bit_true_model_sha256",
            "synthesis_source_sha256",
        }
        synthesis = result["steps"]["synthesise"]
        assert result["success"] is synthesis["success"] is True
        assert synthesis["resources"]["ffs"] > 0

    def test_two_different_networks_give_different_hardware(self) -> None:
        first = run_pipeline(self._supported_graph())
        graph = self._supported_graph()
        cast(list[dict[str, Any]], graph["projections"])[0]["weight"] = 0.25
        second = run_pipeline(graph)
        for key in ("input_sha256", "rtl_sha256", "synthesis_source_sha256"):
            assert first["trace"][key] != second["trace"][key], key

    def test_the_format_is_the_callers_choice_among_two(self) -> None:
        result = run_pipeline(self._supported_graph(), q_format="Q16.16")
        assert result["steps"]["lower"]["q_format"] == "Q16.16"
        with pytest.raises(ValueError, match="q_format must be one of"):
            run_pipeline(self._supported_graph(), q_format="Q4.4")

    def test_pipeline_empty_graph(self) -> None:
        result = run_pipeline({"populations": [], "projections": []})
        assert result["success"] is False
        assert result["step"] == "validate"

    def test_pipeline_target(self) -> None:
        result = run_pipeline(self._make_graph(), target="ecp5")
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

    def test_a_graph_it_cannot_lower_is_refused_with_every_reason(self) -> None:
        """No stand-in is synthesised: the step names what the hardware cannot be."""
        result = run_pipeline(self._make_graph())

        assert result["success"] is False
        assert result["step"] == "lower"
        assert result["error"] == "the graph cannot be lowered to hardware exactly"
        assert len(result["reasons"]) == 2
        assert all("model AdExNeuron has no hardware lowering" in r for r in result["reasons"])
        assert "synthesise" not in result["steps"]

    def test_the_refusal_leaks_nothing_about_the_host(self) -> None:
        text = str(run_pipeline(self._make_graph()))
        assert "Traceback" not in text
        assert "/home/" not in text
        assert "/media/" not in text

    def test_the_simulation_it_did_run_is_still_reported(self) -> None:
        result = run_pipeline(self._make_graph())
        assert result["steps"]["validate"] == {"passed": True}
        assert "n_spikes" in result["steps"]["simulate"]

    def test_a_network_the_compiler_cannot_hold_stops_at_compile(self) -> None:
        """1050 neurons fully connected exceed the compiler's synapse bound."""
        pop = create_population(
            count=1050, model="PerfectIntegratorNeuron", params={"c_m": 1.0, "v_threshold": 1.0}
        )
        proj = create_projection(pop["id"], pop["id"], weight=0.5, rule="all_to_all")
        graph = {"populations": [pop], "projections": [proj], "duration": 2.0, "dt": 1.0}

        result = run_pipeline(graph)

        assert result["step"] == "compile"
        assert result["error"].startswith("the network compiler refused the lowered graph:")
        assert "lower" in result["steps"]

    def test_without_simulators_the_pipeline_stops_before_synthesis(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty PATH is an installation without Icarus Verilog or a C compiler."""
        monkeypatch.setenv("PATH", str(tmp_path))
        result = run_pipeline(self._supported_graph())
        assert result["step"] == "cosimulate"
        assert "needs iverilog, vvp, gcc" in result["error"]
        assert "synthesise" not in result["steps"]

    def test_rtl_that_is_not_its_model_is_never_synthesised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fault injection: the compiler's real output with one weight altered.

        The only way to reach this guard is a compiler that emits the wrong
        network, which no correct build does; the injected RTL stands in for
        that defect, and everything around it is real.
        """
        import dataclasses

        from sc_neurocore.studio import network_hardware_cosim

        real = network_hardware_cosim.compile_lowered

        def miscompiled(lowered: Any) -> Any:
            result = real(lowered)
            return dataclasses.replace(
                result, top_module=result.top_module.replace("sh000000080", "sh000000040")
            )

        monkeypatch.setattr(network_hardware_cosim, "compile_lowered", miscompiled)
        result = run_pipeline(self._supported_graph())

        assert result["step"] == "cosimulate"
        assert result["steps"]["cosimulate"]["rtl_matches_bit_true_model"] is False
        assert "does not reproduce its bit-true model" in result["error"]
        assert "synthesise" not in result["steps"]
