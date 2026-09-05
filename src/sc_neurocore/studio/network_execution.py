# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio graph lowering to the public Network runtime

"""Lower a resolved Studio graph to public ``Network`` objects and run it.

The lowering builds exactly the objects a user script would build from the
public facade (:class:`~sc_neurocore.network.Population` with the validated
constructor keywords, :class:`~sc_neurocore.network.Projection` with the
stated rule, seed, signed weight and delay in whole steps,
:class:`~sc_neurocore.network.SpikeMonitor` per population and one public
stimulus per driven population) and runs the reference Python loop. The
result reports the executed specification, the execution semantics of that
loop (previous-step spike propagation, delay buffering, the stimulus
timestep), a topology artefact with the CSR digest of every projection and
every spike event of every population. The Rust network runner is recorded as
rejected: it constructs populations with default parameters and carries no
stimuli, so it cannot preserve the graph.
"""

from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass
from typing import Any

import numpy as np

import sc_neurocore
from sc_neurocore.network import (
    Network,
    PoissonInput,
    Population,
    Projection,
    SpikeMonitor,
    StepCurrent,
)
from sc_neurocore.network.topology import all_to_all, random_connectivity
from sc_neurocore.studio.analysis_contract import (
    MODEL_DEFINED_UNIT,
    MetricContract,
    attach_contract,
)
from sc_neurocore.studio.model_run_contract import bounded_diagnostic
from sc_neurocore.studio.network_graph_spec import GraphSpec, PopulationSpec, ProjectionSpec

GRAPH_RESULT_SCHEMA_VERSION = "studio.network-graph-result.v1"
PROJECTION_LATENCY_STEPS = 1
CONNECTIVITY_ELEMENT_BUDGET = 200_000
RATE_BIN_COUNT = 100

_RUST_REJECTION = {
    "name": "rust-network-runner",
    "reason": (
        "constructs populations with default model parameters, carries no stimuli "
        "and reads only the maximum delay; it cannot preserve this graph"
    ),
}


class GraphExecutionFailure(RuntimeError):
    """Raised when a lowered graph fails while running.

    Parameters
    ----------
    reason : str
        Bounded, path-free description (exception class and message, or the
        population whose state is non-finite).
    """

    def __init__(self, *, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

    def to_public_detail(self) -> dict[str, object]:
        """Return the path-free public error detail."""
        return {"error": "graph_execution_failed", "reason": self.reason}


@dataclass(frozen=True, slots=True)
class LoweredProjection:
    """One public projection with its CSR arrays and the autapses removed."""

    spec: ProjectionSpec
    projection: Projection
    autapses_removed: int

    @property
    def csr_sha256(self) -> str:
        """Digest of the CSR arrays (indptr, indices, data) as int64/int64/float64 bytes."""
        return csr_digest(self.projection.indptr, self.projection.indices, self.projection.data)


@dataclass(frozen=True, slots=True)
class LoweredGraph:
    """The public objects of one resolved graph, ready for ``Network.run``."""

    spec: GraphSpec
    populations: tuple[Population, ...]
    monitors: tuple[SpikeMonitor, ...]
    projections: tuple[LoweredProjection, ...]
    stimuli: tuple[StepCurrent | PoissonInput, ...]
    network: Network

    @property
    def network_dt_s(self) -> float:
        """Timestep handed to ``Network.run`` (seconds; scales stimuli only)."""
        return self.spec.dt / 1000.0

    @property
    def n_synapses(self) -> int:
        """Total synapses across every projection."""
        return sum(lowered.projection.n_synapses for lowered in self.projections)


def csr_digest(
    indptr: np.ndarray[Any, Any], indices: np.ndarray[Any, Any], data: np.ndarray[Any, Any]
) -> str:
    """Return the SHA-256 of the CSR arrays in their canonical dtypes."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(indptr, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(indices, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(data, dtype=np.float64).tobytes())
    return digest.hexdigest()


def connectivity_arrays(
    spec: ProjectionSpec, n_source: int, n_target: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], int]:
    """Return the CSR arrays of ``spec`` and the number of autapses removed.

    ``random`` uses the public Erdős–Rényi generator with the projection seed;
    ``all_to_all`` the public full generator. On a self-projection without
    ``autapses`` the diagonal entries are removed; nothing else is added or
    dropped.
    """
    if spec.rule == "all_to_all":
        indptr, indices, data = all_to_all(n_source, n_target, spec.weight)
    else:
        indptr, indices, data = random_connectivity(
            n_source, n_target, spec.probability, spec.weight, spec.seed
        )
    removed = 0
    if spec.source == spec.target and not spec.autapses:
        rows = np.repeat(np.arange(n_source, dtype=np.int64), np.diff(indptr))
        keep = rows != indices
        removed = int(np.count_nonzero(~keep))
        if removed:
            new_indptr = np.zeros(n_source + 1, dtype=np.int64)
            np.cumsum(np.bincount(rows[keep], minlength=n_source), out=new_indptr[1:])
            indptr, indices, data = new_indptr, indices[keep], data[keep]
    return indptr, indices, data, removed


def lower_graph(spec: GraphSpec) -> LoweredGraph:
    """Build the public network objects of ``spec`` without running them."""
    populations: dict[str, Population] = {}
    ordered: list[Population] = []
    monitors: list[SpikeMonitor] = []
    stimuli: list[StepCurrent | PoissonInput] = []
    dt_s = spec.dt / 1000.0
    for population in spec.populations:
        public = Population(
            population.inputs.cls,
            population.count,
            params=dict(population.inputs.constructor_kwargs),
            label=population.id,
        )
        populations[population.id] = public
        ordered.append(public)
        monitors.append(SpikeMonitor(public, label=f"spikes_{population.id}"))
        stimulus = _stimulus(population, spec.n_steps, dt_s)
        if stimulus is not None:
            stimulus.target = public
            stimuli.append(stimulus)
    lowered: list[LoweredProjection] = []
    for projection in spec.projections:
        source = populations[projection.source]
        target = populations[projection.target]
        indptr, indices, data, removed = connectivity_arrays(projection, source.n, target.n)
        public_projection = Projection(
            source,
            target,
            weight=projection.weight,
            probability=projection.probability,
            delay=float(projection.delay_steps),
            topology=(indptr, indices, data),
            seed=projection.seed,
        )
        lowered.append(LoweredProjection(projection, public_projection, removed))
    network = Network(seed=spec.seed)
    for public in ordered:
        network.add(public)
    for item in lowered:
        network.add(item.projection)
    for monitor in monitors:
        network.add(monitor)
    for stimulus in stimuli:
        network.add(stimulus)
    return LoweredGraph(
        spec=spec,
        populations=tuple(ordered),
        monitors=tuple(monitors),
        projections=tuple(lowered),
        stimuli=tuple(stimuli),
        network=network,
    )


def _stimulus(
    population: PopulationSpec, n_steps: int, dt_s: float
) -> StepCurrent | PoissonInput | None:
    drive = population.drive
    if drive.kind == "constant":
        assert drive.current is not None
        return StepCurrent(0, n_steps, drive.current)
    if drive.kind == "poisson":
        assert drive.rate_hz is not None and drive.weight is not None and drive.seed is not None
        return PoissonInput(population.count, drive.rate_hz, drive.weight, dt=dt_s, seed=drive.seed)
    return None


def run_lowered_graph(lowered: LoweredGraph) -> None:
    """Run the reference Python loop; report a raising step or non-finite state.

    Raises
    ------
    GraphExecutionFailure
        When a neuron step raises or a population ends with a non-finite
        membrane voltage.
    """
    spec = lowered.spec
    dt_s = lowered.network_dt_s
    try:
        lowered.network.run(duration=spec.n_steps * dt_s, dt=dt_s, backend="python")
    except (ArithmeticError, ValueError, TypeError) as exc:
        raise GraphExecutionFailure(reason=bounded_diagnostic(exc)) from exc
    for population, public in zip(spec.populations, lowered.populations, strict=True):
        if not bool(np.all(np.isfinite(public.voltages))):
            raise GraphExecutionFailure(
                reason=f"population {population.id} ended with a non-finite membrane voltage"
            )


def _rate_bins(steps: np.ndarray[Any, Any], count: int, n_steps: int, dt: float) -> dict[str, Any]:
    bin_steps = max(1, n_steps // RATE_BIN_COUNT)
    n_bins = n_steps // bin_steps
    counts = np.bincount(steps // bin_steps, minlength=n_bins)[:n_bins]
    bin_s = bin_steps * dt / 1000.0
    rates = counts.astype(np.float64) / (count * bin_s)
    return {
        "bin_steps": bin_steps,
        "bin_ms": bin_steps * dt,
        "time_ms": (np.arange(n_bins) * bin_steps * dt).tolist(),
        "rate_hz": rates.tolist(),
        "covered_steps": int(n_bins * bin_steps),
    }


def _population_result(
    population: PopulationSpec, monitor: SpikeMonitor, n_steps: int, dt: float
) -> dict[str, Any]:
    steps, neurons = monitor.raster_data()
    order = np.lexsort((neurons, steps))
    steps = steps[order]
    neurons = neurons[order]
    duration_s = n_steps * dt / 1000.0
    return {
        "id": population.id,
        "label": population.label,
        "model": population.model,
        "count": population.count,
        "neuron_type": population.neuron_type,
        "offset": population.offset,
        "n_spikes": int(steps.size),
        "mean_rate_hz": float(steps.size / (population.count * duration_s)),
        "events": {"step": steps.tolist(), "neuron": neurons.tolist()},
        "rate": _rate_bins(steps, population.count, n_steps, dt),
    }


def _topology_block(lowered: LoweredGraph) -> dict[str, Any]:
    n_synapses = lowered.n_synapses
    include = n_synapses <= CONNECTIVITY_ELEMENT_BUDGET
    projections: list[dict[str, Any]] = []
    for item in lowered.projections:
        block: dict[str, Any] = {
            "id": item.spec.id,
            "source": item.spec.source,
            "target": item.spec.target,
            "rule": item.spec.rule,
            "seed": item.spec.seed,
            "weight": item.spec.weight,
            "delay_steps": item.spec.delay_steps,
            "delay_mode": item.projection.delay_mode,
            "n_synapses": item.projection.n_synapses,
            "autapses_removed": item.autapses_removed,
            "csr_sha256": item.csr_sha256,
        }
        if include:
            block["indptr"] = item.projection.indptr.tolist()
            block["indices"] = item.projection.indices.tolist()
        projections.append(block)
    return {
        "n_synapses": n_synapses,
        "connectivity_included": include,
        "connectivity_omitted_reason": None
        if include
        else f"{n_synapses} synapses exceed the {CONNECTIVITY_ELEMENT_BUDGET}-element budget; digests only",
        "projections": projections,
    }


def _execution_block(lowered: LoweredGraph) -> dict[str, Any]:
    return {
        "backend": {"selected": "python", "rejected": [dict(_RUST_REJECTION)]},
        "loop": "public Network._run_python",
        "step_order": (
            "stimuli into currents; projections propagate the previous step's spikes "
            "(delay buffers in whole steps); every population steps once; monitors record"
        ),
        "projection_latency_steps": PROJECTION_LATENCY_STEPS,
        "delay_semantics": (
            "a spike at step t reaches its targets at step t + 1 + delay_steps; "
            "delay 0 means the inherent one-step latency only"
        ),
        "synapse_semantics": (
            "each source spike injects the synaptic weight as drive into the target "
            "for exactly one step; the membrane increment per spike is model-defined"
        ),
        "network_dt_s": lowered.network_dt_s,
        "population_construction": "model class with the validated constructor keywords",
        "autapses": "removed on self-projections unless the projection declares autapses",
        "state_check": "final membrane voltages must be finite; no per-step state trace",
        "runtime": {
            "package_version": sc_neurocore.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def _contract(spec: GraphSpec) -> MetricContract:
    return MetricContract(
        kind="network-population-activity",
        definition=(
            "Per population, every spike event (step, neuron) recorded by a public "
            "SpikeMonitor; mean_rate_hz = spikes / (count × effective duration); binned "
            "rate_hz = spikes in bin / (count × bin duration)"
        ),
        units={
            "events.step": "step index (time = step × dt ms)",
            "mean_rate_hz": "Hz per neuron",
            "rate.rate_hz": "Hz per neuron",
            "rate.time_ms": "ms (bin start)",
            "spike_times": "ms",
            "weight": MODEL_DEFINED_UNIT,
            "drive.current": MODEL_DEFINED_UNIT,
        },
        applicability=(
            "populations of catalogue models with a float drive and no seed field",
            "public Network Python loop with previous-step spike propagation",
            f"dt {spec.dt:g} ms shared by every population",
        ),
        limitations=(
            "binned rates cover only whole bins; the remainder of the run is counted "
            "in mean_rate_hz and events only",
            "membrane traces are not recorded; spikes and final finiteness only",
        ),
        domain="complete",
    )


def graph_result(lowered: LoweredGraph) -> dict[str, Any]:
    """Assemble the public result of a run graph."""
    spec = lowered.spec
    populations = [
        _population_result(population, monitor, spec.n_steps, spec.dt)
        for population, monitor in zip(spec.populations, lowered.monitors, strict=True)
    ]
    spike_steps: list[np.ndarray[Any, Any]] = []
    spike_neurons: list[np.ndarray[Any, Any]] = []
    for population, block in zip(spec.populations, populations, strict=True):
        steps = np.asarray(block["events"]["step"], dtype=np.int64)
        neurons = np.asarray(block["events"]["neuron"], dtype=np.int64) + population.offset
        spike_steps.append(steps)
        spike_neurons.append(neurons)
    all_steps = np.concatenate(spike_steps) if spike_steps else np.zeros(0, dtype=np.int64)
    all_neurons = np.concatenate(spike_neurons) if spike_neurons else np.zeros(0, dtype=np.int64)
    order = np.lexsort((all_neurons, all_steps))
    all_steps = all_steps[order]
    all_neurons = all_neurons[order]
    n_spikes = int(all_steps.size)
    payload: dict[str, Any] = {
        "success": True,
        "schema_version": GRAPH_RESULT_SCHEMA_VERSION,
        "spec": spec.to_public_dict(),
        "execution": _execution_block(lowered),
        "populations": populations,
        "topology": _topology_block(lowered),
        "n_total": spec.n_neurons,
        "n_spikes": n_spikes,
        "spike_times": (all_steps * spec.dt).tolist(),
        "spike_neurons": all_neurons.tolist(),
        "duration": spec.n_steps * spec.dt,
        "dt": spec.dt,
        "n_steps": spec.n_steps,
        "graph_summary": {
            "n_populations": len(spec.populations),
            "n_projections": len(spec.projections),
            "n_neurons": spec.n_neurons,
            "n_synapses": lowered.n_synapses,
            "n_excitatory": sum(p.count for p in spec.populations if p.neuron_type == "excitatory"),
            "n_inhibitory": sum(p.count for p in spec.populations if p.neuron_type == "inhibitory"),
        },
    }
    return attach_contract(payload, _contract(spec))


def simulate_graph_spec(spec: GraphSpec) -> dict[str, Any]:
    """Lower, run and report one resolved graph.

    Raises
    ------
    GraphExecutionFailure
        From :func:`run_lowered_graph`.
    """
    lowered = lower_graph(spec)
    run_lowered_graph(lowered)
    return graph_result(lowered)


__all__ = [
    "CONNECTIVITY_ELEMENT_BUDGET",
    "GRAPH_RESULT_SCHEMA_VERSION",
    "PROJECTION_LATENCY_STEPS",
    "GraphExecutionFailure",
    "LoweredGraph",
    "LoweredProjection",
    "connectivity_arrays",
    "csr_digest",
    "graph_result",
    "lower_graph",
    "run_lowered_graph",
    "simulate_graph_spec",
]
