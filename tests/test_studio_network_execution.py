# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio graph execution against the public Network runtime

"""The Studio graph runs exactly what a public ``Network`` script would run."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import sc_neurocore.neurons.models as models
from sc_neurocore.network import Network, PoissonInput, Population, Projection, SpikeMonitor
from sc_neurocore.network import StepCurrent
from sc_neurocore.network.topology import random_connectivity
from sc_neurocore.studio.network_execution import (
    GraphExecutionFailure,
    connectivity_arrays,
    csr_digest,
    lower_graph,
    run_lowered_graph,
    simulate_graph_spec,
)
from sc_neurocore.studio.network_graph import create_population, create_projection, simulate_graph
from sc_neurocore.studio.network_graph_spec import resolve_graph

DT = 0.1
# SCLapicqueLIFNeuron defaults: v_rest 0, v_reset 0, v_threshold 1, tau 20, R 1.
# Under constant drive I the exact flow from reset reaches threshold after
# k = ceil(tau/dt · ln(I / (I − 1))) steps; the spike is recorded at step k − 1.


def _period_steps(current: float, tau: float = 20.0) -> int:
    return math.ceil(tau / DT * math.log(current / (current - 1.0)))


def _expected_spike_steps(current: float, n_steps: int, tau: float = 20.0) -> list[int]:
    period = _period_steps(current, tau)
    return [step for step in range(period - 1, n_steps, period)]


def _pop(pid: str, count: int, neuron_type: str = "excitatory", **kw: Any) -> dict[str, Any]:
    pop = create_population(label=pid, count=count, neuron_type=neuron_type, **kw)
    pop["id"] = pid
    return pop


def _proj(pid: str, source: str, target: str, **kw: Any) -> dict[str, Any]:
    proj = create_projection(source, target, **kw)
    proj["id"] = pid
    return proj


class TestAnalyticControls:
    def test_constant_drive_spikes_at_the_exact_flow_period(self) -> None:
        graph = {
            "populations": [_pop("e", 3, drive={"kind": "constant", "current": 1.2})],
            "projections": [],
            "duration": 200.0,
            "dt": DT,
        }
        result = simulate_graph(graph)
        expected = _expected_spike_steps(1.2, 2000)
        assert expected == [358, 717, 1076, 1435, 1794]
        events = result["populations"][0]["events"]
        assert events["step"] == sorted(expected * 3)
        assert result["n_spikes"] == 15
        assert result["spike_times"][:3] == [358 * DT] * 3
        assert result["spike_neurons"][:3] == [0, 1, 2]

    def test_changing_tau_changes_the_result(self) -> None:
        graph = {
            "populations": [
                _pop("e", 1, drive={"kind": "constant", "current": 1.2}, params={"tau": 10.0})
            ],
            "projections": [],
            "duration": 200.0,
            "dt": DT,
        }
        result = simulate_graph(graph)
        expected = _expected_spike_steps(1.2, 2000, tau=10.0)
        assert result["populations"][0]["events"]["step"] == expected
        assert len(expected) == 11
        assert result["spec"]["populations"][0]["parameters"]["tau"] == 10.0
        assert result["spec"]["populations"][0]["overrides_applied"] == ["tau"]

    def test_delay_shifts_arrival_by_exactly_delay_plus_one_step(self) -> None:
        for delay_ms, expected_offset in ((0.0, 1), (0.2, 3), (1.0, 11)):
            graph = {
                "populations": [
                    _pop("e", 1, drive={"kind": "constant", "current": 1.2}),
                    _pop("t", 1, "inhibitory"),
                ],
                "projections": [
                    _proj("et", "e", "t", weight=300.0, rule="all_to_all", delay=delay_ms)
                ],
                "duration": 40.0,
                "dt": DT,
            }
            result = simulate_graph(graph)
            source_steps = result["populations"][0]["events"]["step"]
            target_steps = result["populations"][1]["events"]["step"]
            assert source_steps == [358]
            assert target_steps == [358 + expected_offset], delay_ms
            assert result["execution"]["projection_latency_steps"] == 1

    def test_disconnected_and_undriven_populations_stay_silent(self) -> None:
        graph = {
            "populations": [
                _pop("e", 4, drive={"kind": "constant", "current": 1.2}),
                _pop("lonely", 4, "inhibitory"),
            ],
            "projections": [],
            "duration": 100.0,
            "dt": DT,
        }
        result = simulate_graph(graph)
        assert result["populations"][0]["n_spikes"] == 8
        assert result["populations"][1]["n_spikes"] == 0
        assert result["populations"][1]["mean_rate_hz"] == 0.0

    def test_two_projections_between_the_same_pair_sum_their_currents(self) -> None:
        def run(edges: list[dict[str, Any]]) -> list[int]:
            graph = {
                "populations": [
                    _pop("e", 1, drive={"kind": "constant", "current": 1.2}),
                    _pop("t", 1, "inhibitory"),
                ],
                "projections": edges,
                "duration": 40.0,
                "dt": DT,
            }
            return list(simulate_graph(graph)["populations"][1]["events"]["step"])

        one = _proj("a", "e", "t", weight=150.0, rule="all_to_all")
        two = _proj("b", "e", "t", weight=150.0, rule="all_to_all")
        assert run([one]) == []
        assert run([one, two]) == [359]

    def test_inhibitory_weight_is_subtracted_not_absoluted(self) -> None:
        graph = {
            "populations": [
                _pop("e", 1, drive={"kind": "constant", "current": 1.2}),
                _pop("i", 1, "inhibitory", drive={"kind": "constant", "current": 1.2}),
                _pop("t", 1, "inhibitory"),
            ],
            "projections": [
                _proj("et", "e", "t", weight=300.0, rule="all_to_all"),
                _proj("it", "i", "t", weight=-300.0, rule="all_to_all"),
            ],
            "duration": 40.0,
            "dt": DT,
        }
        result = simulate_graph(graph)
        assert result["populations"][0]["events"]["step"] == [358]
        assert result["populations"][1]["events"]["step"] == [358]
        assert result["populations"][2]["events"]["step"] == []
        assert result["spec"]["projections"][1]["sign"] == "inhibitory"


class TestTopology:
    def test_probability_and_seed_determine_the_connectivity(self) -> None:
        def synapses(probability: float, seed: int | None = None) -> tuple[int, str]:
            edge = _proj("ee", "e", "e", weight=1.0, probability=probability)
            if seed is not None:
                edge["seed"] = seed
            spec = resolve_graph(
                {"populations": [_pop("e", 50)], "projections": [edge], "duration": 1.0}
            )
            lowered = lower_graph(spec)
            item = lowered.projections[0]
            return item.projection.n_synapses, item.csr_sha256

        low, low_digest = synapses(0.1)
        high, high_digest = synapses(0.5)
        assert 0 < low < high < 50 * 49
        assert low_digest != high_digest
        assert synapses(0.1) == (low, low_digest)
        assert synapses(0.1, seed=5) != synapses(0.1, seed=6)

    def test_self_projection_removes_autapses_unless_declared(self) -> None:
        graph: dict[str, Any] = {
            "populations": [_pop("e", 30)],
            "projections": [_proj("loop", "e", "e", weight=1.0, rule="all_to_all")],
            "duration": 1.0,
        }
        lowered = lower_graph(resolve_graph(graph))
        item = lowered.projections[0]
        assert item.autapses_removed == 30
        assert item.projection.n_synapses == 30 * 29
        rows = np.repeat(np.arange(30), np.diff(item.projection.indptr))
        assert not np.any(rows == item.projection.indices)
        graph["projections"][0]["autapses"] = True
        kept = lower_graph(resolve_graph(graph)).projections[0]
        assert kept.autapses_removed == 0
        assert kept.projection.n_synapses == 900

    def test_duplicate_edges_draw_independent_derived_seeds(self) -> None:
        graph = {
            "populations": [_pop("e", 40), _pop("i", 10, "inhibitory")],
            "projections": [
                _proj("a", "e", "i", weight=1.0, probability=0.3),
                _proj("b", "e", "i", weight=1.0, probability=0.3),
            ],
            "duration": 1.0,
        }
        lowered = lower_graph(resolve_graph(graph))
        first, second = lowered.projections
        assert first.spec.seed != second.spec.seed
        assert first.csr_sha256 != second.csr_sha256

    def test_topology_artefact_carries_csr_arrays_and_digests(self) -> None:
        graph = {
            "populations": [_pop("e", 20), _pop("i", 5, "inhibitory")],
            "projections": [_proj("ei", "e", "i", weight=2.0, probability=0.5, delay=0.3)],
            "duration": 1.0,
        }
        result = simulate_graph(graph)
        block = result["topology"]["projections"][0]
        assert result["topology"]["connectivity_included"] is True
        assert block["delay_steps"] == 3 and block["delay_mode"] == "uniform"
        assert len(block["indptr"]) == 21
        assert len(block["indices"]) == block["n_synapses"]
        indptr, indices, data = random_connectivity(20, 5, 0.5, 2.0, block["seed"])
        assert block["indptr"] == indptr.tolist()
        assert block["indices"] == indices.tolist()
        assert block["csr_sha256"] == csr_digest(indptr, indices, data)


class TestPublicRuntimeIdentity:
    def test_graph_events_equal_a_direct_public_network_script(self) -> None:
        graph = {
            "populations": [
                _pop("e1", 30, drive={"kind": "constant", "current": 2.0}),
                _pop("e2", 20, drive={"kind": "poisson", "rate_hz": 400.0, "weight": 30.0}),
                _pop("i", 10, "inhibitory", params={"tau": 15.0}),
            ],
            "projections": [
                _proj("a", "e1", "i", weight=60.0, probability=0.4, delay=0.5),
                _proj("b", "e2", "i", weight=60.0, rule="all_to_all"),
                _proj("c", "i", "e1", weight=-20.0, probability=0.3, delay=0.2),
                _proj("d", "e1", "e1", weight=20.0, probability=0.1),
            ],
            "duration": 60.0,
            "dt": DT,
            "seed": 11,
        }
        result = simulate_graph(graph)
        spec = result["spec"]
        assert result["n_spikes"] > 0
        assert all(block["n_spikes"] > 0 for block in result["populations"])

        # The same objects, built by hand from the public facade and the spec.
        dt_s = DT / 1000.0
        cls = models.SCLapicqueLIFNeuron
        e1 = Population(cls, 30, {"dt": DT}, label="e1")
        e2 = Population(cls, 20, {"dt": DT}, label="e2")
        i = Population(cls, 10, {"tau": 15.0, "dt": DT}, label="i")
        pops = {"e1": e1, "e2": e2, "i": i}
        monitors = [SpikeMonitor(e1), SpikeMonitor(e2), SpikeMonitor(i)]
        drive_e1 = StepCurrent(0, 600, 2.0)
        drive_e1.target = e1
        drive_e2 = PoissonInput(
            20, 400.0, 30.0, dt=dt_s, seed=spec["populations"][1]["drive"]["seed"]
        )
        drive_e2.target = e2
        projections = []
        for block in spec["projections"]:
            source = pops[block["source"]]
            target = pops[block["target"]]
            if block["rule"] == "all_to_all":
                indptr = np.repeat(np.arange(source.n + 1) * target.n, 1)
                indices = np.tile(np.arange(target.n), source.n)
                data = np.full(source.n * target.n, block["weight"])
            else:
                indptr, indices, data = random_connectivity(
                    source.n, target.n, block["probability"], block["weight"], block["seed"]
                )
            if source is target:
                rows = np.repeat(np.arange(source.n), np.diff(indptr))
                keep = rows != indices
                indptr = np.concatenate(
                    [[0], np.cumsum(np.bincount(rows[keep], minlength=source.n))]
                )
                indices, data = indices[keep], data[keep]
            projections.append(
                Projection(
                    source,
                    target,
                    weight=block["weight"],
                    delay=float(block["delay_steps"]),
                    topology=(np.asarray(indptr), np.asarray(indices), np.asarray(data)),
                )
            )
        net = Network(e1, e2, i, *projections, *monitors, drive_e1, drive_e2, seed=11)
        net.run(duration=600 * dt_s, dt=dt_s, backend="python")

        for block, monitor in zip(result["populations"], monitors, strict=True):
            steps, neurons = monitor.raster_data()
            order = np.lexsort((neurons, steps))
            assert block["events"]["step"] == steps[order].tolist()
            assert block["events"]["neuron"] == neurons[order].tolist()
        for block, projection in zip(result["topology"]["projections"], projections, strict=True):
            assert block["csr_sha256"] == csr_digest(
                projection.indptr, projection.indices, projection.data
            )


class TestCustody:
    def test_payload_reports_semantics_rates_and_offsets(self) -> None:
        graph = {
            "populations": [
                _pop("e", 4, drive={"kind": "constant", "current": 1.2}),
                _pop("i", 2, "inhibitory", drive={"kind": "constant", "current": 1.2}),
            ],
            "projections": [],
            "duration": 100.0,
            "dt": DT,
        }
        result = simulate_graph(graph)
        assert result["execution"]["backend"]["rejected"][0]["name"] == "rust-network-runner"
        assert result["execution"]["network_dt_s"] == pytest.approx(1e-4)
        assert result["populations"][1]["offset"] == 4
        assert min(result["spike_neurons"][-2:]) >= 4
        e = result["populations"][0]
        assert e["mean_rate_hz"] == pytest.approx(e["n_spikes"] / (4 * 0.1))
        rate = e["rate"]
        assert rate["bin_steps"] == 10 and rate["covered_steps"] == 1000
        assert len(rate["time_ms"]) == len(rate["rate_hz"]) == 100
        counts = np.asarray(rate["rate_hz"]) * 4 * rate["bin_ms"] / 1000.0
        assert int(round(counts.sum())) == e["n_spikes"]
        assert result["contract"]["domain"] == "complete"
        assert result["contract"]["units"]["mean_rate_hz"] == "Hz per neuron"
        assert result["spec"]["graph_sha256"]

    def test_execution_failures_are_reported_not_zeroed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = resolve_graph({"populations": [_pop("e", 2)], "projections": [], "duration": 1.0})
        lowered = lower_graph(spec)

        def poison(self: Network, **_: Any) -> None:
            lowered.populations[0]._voltages[:] = np.nan

        monkeypatch.setattr(Network, "run", poison)
        with pytest.raises(GraphExecutionFailure, match="non-finite membrane voltage"):
            run_lowered_graph(lowered)

        def explode(self: Network, **_: Any) -> None:
            raise ArithmeticError("overflow in step")

        monkeypatch.setattr(Network, "run", explode)
        with pytest.raises(GraphExecutionFailure) as info:
            simulate_graph_spec(spec)
        assert info.value.to_public_detail() == {
            "error": "graph_execution_failed",
            "reason": "ArithmeticError: overflow in step",
        }

    def test_connectivity_arrays_all_to_all_matches_public_generator(self) -> None:
        spec = resolve_graph(
            {
                "populations": [_pop("e", 3), _pop("i", 2, "inhibitory")],
                "projections": [_proj("ei", "e", "i", weight=1.5, rule="all_to_all")],
                "duration": 1.0,
            }
        )
        indptr, indices, data, removed = connectivity_arrays(spec.projections[0], 3, 2)
        assert indptr.tolist() == [0, 2, 4, 6]
        assert indices.tolist() == [0, 1, 0, 1, 0, 1]
        assert data.tolist() == [1.5] * 6
        assert removed == 0
