# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio complete-state and raw-result custody (public Python surface)

"""Complete state and raw-result custody of Studio runs.

Every case fails on the former behaviour, which chose state variables by name
heuristics (recording AdEx parameters ``a``/``b`` as state and dropping the
adaptive threshold ``theta``), labelled the post-step sample ``t`` with
``t*dt``, returned the decimated plot samples as the result and kept no
initial or final snapshot.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from sc_neurocore.neurons.models.adaptive_threshold_if import AdaptiveThresholdIFNeuron
from sc_neurocore.studio import model_simulate
from sc_neurocore.studio.model_introspection import _load_class
from sc_neurocore.studio.model_run_contract import ModelSimulationFailure
from sc_neurocore.studio.models import simulate_model
from sc_neurocore.studio.simulation import MAX_PLOT_POINTS, simulate
from sc_neurocore.studio.state_layout import (
    DeclaredState,
    StateLayout,
    StateObservationError,
    declared_state,
    equation_state,
    observe_layout,
    read_variable,
    snapshot,
)

ATIF = "AdaptiveThresholdIFNeuron"
ATIF_OVERRIDES = {"delta_theta": 8.0, "tau_theta": 30.0, "theta_rest": -48.0}

REPRESENTATIVES: dict[str, tuple[str, ...]] = {
    # adaptive threshold, multi-state conductance, map, population rate,
    # stochastic, multi-compartment
    ATIF: ("v", "theta"),
    "HodgkinHuxleyNeuron": ("v", "m", "h", "n"),
    "RulkovMapNeuron": ("x", "y"),
    "JansenRitUnit": ("y0", "y3", "y1", "y4", "y2", "y5"),
    "PoissonNeuron": ("rng_state",),
    "DendrifyNeuron": ("v_s", "v_d", "d_timer"),
}


def _run(name: str, **kwargs: Any) -> dict[str, Any]:
    return simulate_model(name, use_fast_path=False, **kwargs)


class TestDeclaredLayout:
    @pytest.mark.parametrize(("name", "expected"), sorted(REPRESENTATIVES.items()))
    def test_layout_follows_the_descriptor_not_name_heuristics(
        self, name: str, expected: tuple[str, ...]
    ) -> None:
        source, _stem, declared = declared_state(name)
        assert source == "descriptor"
        assert tuple(spec.name for spec in declared) == expected
        result = _run(name, duration=2.0)
        layout = result["state_layout"]
        assert [v["name"] for v in layout["variables"]] == list(expected)
        assert layout["recorded"] == list(expected)
        assert set(result["states"]) == set(expected)
        assert set(result["raw"]["states"]) == set(expected)
        assert all(v["kind"] == "scalar" and v["shape"] == [] for v in layout["variables"])
        assert all(
            v["role"] in {"biological", "auxiliary", "unassigned"} for v in layout["variables"]
        )
        assert layout["complete"] is True, layout["incomplete_reasons"]

    def test_parameters_are_not_recorded_as_state(self) -> None:
        result = _run("AdExNeuron", duration=2.0)
        assert list(result["states"]) == ["v", "w"]
        assert "a" not in result["states"] and "b" not in result["states"]

    def test_profile_roles_and_units_are_carried(self) -> None:
        result = _run("LapicqueNeuron", duration=2.0)
        layout = result["state_layout"]
        assert layout["source"] == "descriptor"
        assert layout["schema_profile"] == "lapicque"
        by_name = {v["name"]: v for v in layout["variables"]}
        assert by_name["v"]["role"] == "biological"
        assert by_name["v"]["unit"]
        assert by_name["v"]["declared_init"] == pytest.approx(by_name["v"]["declared_init"])

    def test_vector_state_keeps_its_shape(self) -> None:
        result = _run("AmariNeuralField", duration=2.0)
        by_name = {v["name"]: v for v in result["state_layout"]["variables"]}
        assert by_name["u"]["kind"] == "vector"
        assert by_name["u"]["shape"] == [64]
        assert by_name["u"]["trace"] == "per-step"
        assert "u" not in result["states"]
        assert np.asarray(result["raw"]["vector_states"]["u"]).shape == (result["n_steps"], 64)
        assert len(result["initial_state"]["u"]) == 64
        assert len(result["final_state"]["u"]) == 64
        assert result["state_layout"]["complete"] is True

    def test_undeclared_layout_is_reported_not_guessed(self) -> None:
        # AstrocyteNeuron mirrors the wrapped astrocyte's calcium into ``v``, so
        # it declares no state of its own while ``v`` moves. RallCableNeuron
        # stood here until its compartment vector became declarable.
        result = _run("AstrocyteNeuron", duration=1.0)
        layout = result["state_layout"]
        assert layout["source"] == "undeclared"
        assert layout["variables"] == []
        assert layout["complete"] is False
        assert "v" in layout["undeclared_mutable"]
        assert result["states"] == {}
        assert result["initial_state"] == {} and result["final_state"] == {}

    def test_private_registers_outside_the_declared_layout_make_custody_incomplete(
        self,
    ) -> None:
        result = _run("SRM0Neuron", duration=5.0)
        layout = result["state_layout"]
        assert layout["complete"] is False
        assert any(name.startswith("_") for name in layout["undeclared_mutable"])
        assert any("not declared state" in reason for reason in layout["incomplete_reasons"])

    def test_declared_variable_missing_on_the_instance_is_named(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _run("PoissonNeuron", duration=2.0)
        assert result["state_layout"]["complete"] is True

        # A declaration the instance does not carry is reported, never dropped.
        # No catalogue model declares such a variable any more — every declared
        # name is an attribute, held by
        # tests/test_studio_declared_state_observable.py — so the declaration is
        # injected here. Pinning this against a model that happened to be broken
        # would have tied the guarantee to the defect's lifetime.
        phantom = DeclaredState(
            name="never_carried",
            role="auxiliary",
            unit="ms",
            meaning="declared register the instance does not carry",
            declared_init=0.0,
        )
        real_declared_state = declared_state

        def _with_phantom(name: str) -> tuple[Any, str, tuple[DeclaredState, ...]]:
            source, stem, variables = real_declared_state(name)
            return source, stem, (*variables, phantom)

        monkeypatch.setattr(model_simulate, "declared_state", _with_phantom)
        result = _run("PoissonNeuron", duration=2.0)
        by_name = {v["name"]: v for v in result["state_layout"]["variables"]}
        assert by_name["never_carried"]["observable"] is False
        assert by_name["never_carried"]["reason"] == "not an attribute of the model instance"
        assert "never_carried" not in result["states"]
        assert result["state_layout"]["complete"] is False
        assert {"name": "never_carried", "reason": "not an attribute of the model instance"} in (
            result["effective_inputs"]["state_recording"]["excluded"]
        )


class TestObservationClockAndSnapshots:
    def test_samples_are_post_step_and_snapshots_are_exact(self) -> None:
        result = _run(ATIF, param_overrides=ATIF_OVERRIDES, current=20.0, duration=50.0)
        dt = result["dt"]
        n = result["n_steps"]
        assert result["observation"] == {
            "clock": "post-step",
            "dt": dt,
            "initial_time_ms": 0.0,
            "sample_time_ms": "(index + 1) * dt",
            "drive_interval_ms": "[index * dt, (index + 1) * dt)",
        }
        assert result["time"] == [(i + 1) * dt for i in range(n)]

        reference = AdaptiveThresholdIFNeuron(**ATIF_OVERRIDES)
        assert result["initial_state"] == {"v": reference.v, "theta": reference.theta}
        expected_v: list[float] = []
        expected_theta: list[float] = []
        spikes: list[int] = []
        for t in range(n):
            if reference.step(20.0):
                spikes.append(t)
            expected_v.append(reference.v)
            expected_theta.append(reference.theta)
        assert result["raw"]["states"] == {"v": expected_v, "theta": expected_theta}
        assert result["final_state"] == {"v": reference.v, "theta": reference.theta}
        assert result["spikes"] == spikes
        assert result["raw"]["spike_indices"] == spikes
        assert result["raw"]["spike_times_ms"] == [(t + 1) * dt for t in spikes]

    def test_multi_state_final_snapshot_matches_the_instance(self) -> None:
        result = _run("HodgkinHuxleyNeuron", duration=5.0)
        cls = _load_class("HodgkinHuxleyNeuron")
        reference = cls()
        for sample in result["raw"]["drive"]:
            reference.step(sample)
        assert result["final_state"] == {
            name: getattr(reference, name) for name in ("v", "m", "h", "n")
        }

    def test_equation_playground_shares_the_contract(self) -> None:
        result = simulate(["dv/dt = I"], init={"v": 0.5}, dt=0.1, duration=1.0, current=1.0)
        assert result["state_layout"]["source"] == "equations"
        assert result["state_layout"]["recorded"] == ["v"]
        assert result["initial_state"] == {"v": 0.5}
        assert result["time"][0] == pytest.approx(0.1)
        assert result["raw"]["states"]["v"][0] == pytest.approx(0.6)
        assert result["final_state"]["v"] == pytest.approx(1.5)
        assert result["state_layout"]["complete"] is True

    def test_equation_playground_non_finite_state_is_a_failure_not_a_nan_trace(
        self,
    ) -> None:
        with pytest.raises(ModelSimulationFailure) as info:
            simulate(["dv/dt = v*v"], init={"v": 1e200}, dt=1.0, duration=3.0, current=0.0)
        assert info.value.model == "ode"
        assert info.value.step == 0
        assert "non-finite" in info.value.diagnostic


class TestDisplayProjection:
    @pytest.mark.parametrize("n_steps", [MAX_PLOT_POINTS + 1, 9_999])
    def test_boundaries_keep_the_bound_the_final_sample_and_every_spike_peak(
        self, n_steps: int
    ) -> None:
        result = _run(ATIF, current=20.0, duration=n_steps * 0.1)
        assert result["n_steps"] == n_steps
        display = result["display"]
        assert display["method"] == "bucket-extrema"
        assert display["point_count"] <= MAX_PLOT_POINTS
        assert len(result["time"]) == len(result["states"]["v"]) == display["point_count"]
        index = display["sample_index"]
        assert index[0] == 0 and index[-1] == n_steps - 1
        raw_v = np.asarray(result["raw"]["states"]["v"])
        assert result["states"]["v"] == raw_v[index].tolist()
        assert max(result["states"]["v"]) == raw_v.max()
        assert min(result["states"]["v"]) == raw_v.min()
        assert result["states"]["v"][-1] == result["final_state"]["v"]
        assert len(result["spikes"]) == result["spike_count"] > 0
        assert result["spikes"] == result["raw"]["spike_indices"]
        assert result["effective_inputs"]["display_points"] == display["point_count"]

    def test_short_run_is_shown_in_full(self) -> None:
        result = _run(ATIF, duration=10.0)
        assert result["display"]["method"] == "identity"
        assert result["display"]["sample_index"] == list(range(result["n_steps"]))
        assert result["states"] == result["raw"]["states"]

    def test_equation_playground_boundaries(self) -> None:
        for n_steps in (MAX_PLOT_POINTS + 1, 9_999):
            result = simulate(
                ["dv/dt = I"], init={"v": 0.0}, dt=0.1, duration=n_steps * 0.1, current=1.0
            )
            assert result["n_steps"] == n_steps
            assert len(result["time"]) <= MAX_PLOT_POINTS
            assert result["display"]["sample_index"][-1] == n_steps - 1
            assert len(result["raw"]["states"]["v"]) == n_steps


class TestRawExportAndReplay:
    def test_raw_arrays_survive_json_and_replay_independently(self) -> None:
        result = _run(ATIF, param_overrides=ATIF_OVERRIDES, protocol="sine", duration=50.0)
        exported = json.loads(json.dumps(result, allow_nan=False))
        receipt = exported["effective_inputs"]
        raw = exported["raw"]
        assert raw["included"] is True
        assert len(raw["drive"]) == exported["n_steps"]

        # Independent replay: rebuild the model from the receipt, drive it with
        # the exported raw drive, compare every raw sample and both snapshots.
        cls = _load_class(receipt["model"])
        replay = cls(**receipt["parameters"])
        assert exported["initial_state"] == {"v": replay.v, "theta": replay.theta}
        replayed: dict[str, list[float]] = {"v": [], "theta": []}
        spikes: list[int] = []
        for t, sample in enumerate(raw["drive"]):
            if replay.step(sample):
                spikes.append(t)
            replayed["v"].append(replay.v)
            replayed["theta"].append(replay.theta)
        assert replayed == raw["states"]
        assert spikes == raw["spike_indices"]
        assert exported["final_state"] == {"v": replay.v, "theta": replay.theta}

    def test_raw_block_declares_its_budget(self) -> None:
        result = _run(ATIF, duration=1.0)
        raw = result["raw"]
        assert raw["schema_version"] == "studio.raw-trace.v1"
        assert raw["element_budget"] == 2_000_000
        assert raw["element_count"] == result["n_steps"] * 3
        assert raw["vector_snapshots_only"] == []


class TestRustBackendCustody:
    def test_rust_result_is_never_presented_as_complete_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _batch(
            _name: str, n_steps: int, current: np.ndarray[Any, Any]
        ) -> dict[str, np.ndarray[Any, Any]]:
            return {
                "voltages": np.full(n_steps, -65.0) + current * 0.0,
                "spikes": np.array([3], dtype=np.int64),
            }

        monkeypatch.setattr(model_simulate, "_load_rust_batch_simulate", lambda: _batch)
        result = simulate_model("AdExNeuron", duration=2.0)
        layout = result["state_layout"]
        assert result["effective_inputs"]["backend"] == "rust"
        assert result["initial_state"] is None
        assert result["final_state"] == {"v": -65.0}
        assert layout["complete"] is False
        by_name = {v["name"]: v for v in layout["variables"]}
        assert by_name["w"]["observable"] is False
        assert by_name["w"]["reason"] == "not exported by the Rust batch backend"
        assert "the Rust batch backend exposes no initial snapshot" in layout["custody_notes"]
        assert result["raw"]["states"] == {"v": [-65.0] * 20}
        assert result["time"][0] == pytest.approx(0.1)


class TestLayoutPrimitives:
    def test_observe_layout_reports_each_kind(self) -> None:
        class Model:
            def __init__(self) -> None:
                self.v = -65.0
                self.u = np.zeros(3)
                self.flag = True
                self.name = "x"

        declared = (
            DeclaredState("v", "biological", "mV", "", -65.0),
            DeclaredState("u", "auxiliary", "", "", 0.0),
            DeclaredState("flag", "unassigned", "", "", None),
            DeclaredState("name", "unassigned", "", "", None),
            DeclaredState("absent", "unassigned", "", "", None),
        )
        layout = observe_layout(
            Model(), "descriptor", "stem", declared, n_steps=10, element_budget=100
        )
        by_name = {v.name: v for v in layout.variables}
        assert (by_name["v"].kind, by_name["v"].shape, by_name["v"].trace) == (
            "scalar",
            (),
            "per-step",
        )
        assert (by_name["u"].kind, by_name["u"].shape, by_name["u"].trace) == (
            "vector",
            (3,),
            "per-step",
        )
        # A declared flag is state and reads as the 0.0/1.0 the schema lowers it
        # as; a declared string is not a quantity and still cannot be recorded.
        assert (by_name["flag"].kind, by_name["flag"].shape) == ("scalar", ())
        assert by_name["flag"].observable is True
        assert by_name["name"].observable is False
        assert by_name["absent"].reason == "not an attribute of the model instance"
        assert layout.complete is False

        tight = observe_layout(
            Model(), "descriptor", "stem", declared[:2], n_steps=10, element_budget=20
        )
        assert {v.name: v.trace for v in tight.variables} == {
            "v": "per-step",
            "u": "snapshots-only",
        }
        assert "u: recorded in snapshots only" in tight.incomplete_reasons()

    def test_read_variable_enforces_kind_shape_and_finiteness(self) -> None:
        class Model:
            v: object = -65.0
            u: object = np.zeros(2)

        declared = (
            DeclaredState("v", "unassigned", "", "", None),
            DeclaredState("u", "unassigned", "", "", None),
        )
        layout = observe_layout(Model(), "descriptor", "", declared, n_steps=1, element_budget=10)
        model = Model()
        assert snapshot(model, layout)["v"] == -65.0
        model.v = float("nan")
        with pytest.raises(StateObservationError, match="non-finite"):
            read_variable(model, layout.variables[0])
        model.v = np.zeros(2)
        with pytest.raises(StateObservationError, match="no longer a scalar"):
            read_variable(model, layout.variables[0])
        model.u = np.zeros(3)
        with pytest.raises(StateObservationError, match="changed shape"):
            read_variable(model, layout.variables[1])
        model.u = np.array([1.0, np.inf])
        with pytest.raises(StateObservationError, match="non-finite"):
            read_variable(model, layout.variables[1])

    def test_equation_state_and_public_layout(self) -> None:
        declared = equation_state(["v", "w"], {"v": 1.0})
        assert [(d.name, d.declared_init) for d in declared] == [("v", 1.0), ("w", None)]
        layout = StateLayout(source="equations", schema_profile="", variables=())
        public = layout.to_public_dict()
        assert public["schema_version"] == "studio.state-layout.v1"
        assert public["complete"] is True and public["variables"] == []

    def test_descriptor_state_is_declared_for_every_registered_class(self) -> None:
        from sc_neurocore.neurons.models import _CLASS_TO_MODULE

        undeclared = sorted(
            name for name in _CLASS_TO_MODULE if declared_state(name)[0] == "undeclared"
        )
        # Descriptors that declare no state are reported as undeclared, never
        # guessed. The census is pinned rather than sampled, so the gap shrinks
        # only by a deliberate test change. Two families remain. Four models
        # carry their state under a private name
        # (DISCOVERED-PRIVATE-REGISTERS-OUTSIDE-LAYOUT). The fifth,
        # InhomogeneousPoissonNeuron, mutates no attribute at all and is still
        # here on purpose: it draws from the process-wide NumPy generator, so
        # its run cannot be reproduced from anything recorded and an empty
        # declaration would publish it as complete
        # (DISCOVERED-MODELS-DRAW-FROM-THE-GLOBAL-GENERATOR). The two that left
        # this list, McCullochPittsNeuron and SiegertTransferFunction, now
        # assert `stateless` in their descriptors and are covered by
        # tests/test_stateless_declaration.py.
        assert undeclared == [
            "AstrocyteNeuron",
            "GLMNeuron",
            "GammaRenewalNeuron",
            "HybridFisherPosnerLIFNeuron",
            "InhomogeneousPoissonNeuron",
        ]
        assert declared_state(ATIF)[0] == "descriptor"
