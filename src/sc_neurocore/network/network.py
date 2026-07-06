# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Network: main simulation orchestrator

"""Network: main simulation orchestrator for populations, projections, and monitors."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from .population import Population
from .projection import Projection
from .monitor import SpikeMonitor, StateMonitor, RateMonitor
from .stimulus import TimedArray, PoissonInput, StepCurrent

_RUST_ENGINE: Any = None


def _load_network_runner_class() -> Any:
    try:
        from sc_neurocore_engine.network import get_network_runner_class
    except ImportError:
        try:
            from sc_neurocore_engine import NetworkRunner
        except (AttributeError, ImportError) as exc:
            raise ImportError("sc_neurocore_engine NetworkRunner is unavailable") from exc
        return NetworkRunner
    return get_network_runner_class()


def _get_rust_engine() -> Any:
    global _RUST_ENGINE
    if _RUST_ENGINE is None:
        try:
            _RUST_ENGINE = _load_network_runner_class()
        except ImportError:
            _RUST_ENGINE = False
    return _RUST_ENGINE


def _rust_supports_model(model_name: str) -> bool:
    engine = _get_rust_engine()
    if engine is False:
        return False
    supported = engine.supported_models()
    if model_name in supported:
        return True
    # Try without "Neuron" suffix
    return model_name.endswith("Neuron") and model_name[:-6] in supported


class Network:
    """Declarative network: collects objects, runs the simulation loop."""

    def __init__(self, *objects: Any, seed: int = 42, fim_lambda: float = 0.0) -> None:
        self.populations: list[Population] = []
        self.projections: list[Projection] = []
        self.spike_monitors: list[SpikeMonitor] = []
        self.state_monitors: list[StateMonitor] = []
        self.rate_monitors: list[RateMonitor] = []
        self.stimuli: list[TimedArray | PoissonInput | StepCurrent] = []
        self.seed = seed
        self.fim_lambda = fim_lambda
        self._spike_gating = False
        for obj in objects:
            self.add(obj)

    def add(self, obj: Any) -> None:
        """Register a simulation object by type."""
        if isinstance(obj, Population):
            self.populations.append(obj)
        elif isinstance(obj, Projection):
            self.projections.append(obj)
        elif isinstance(obj, SpikeMonitor):
            self.spike_monitors.append(obj)
        elif isinstance(obj, StateMonitor):
            self.state_monitors.append(obj)
        elif isinstance(obj, RateMonitor):
            self.rate_monitors.append(obj)
        elif isinstance(obj, (TimedArray, PoissonInput, StepCurrent)):
            self.stimuli.append(obj)
        else:
            raise TypeError(f"Unknown object type: {type(obj).__name__}")

    def _can_use_rust(self) -> bool:
        """Return whether this network can preserve semantics on the Rust backend."""
        if self.stimuli:
            return False
        if self._rust_incompatibilities():
            return False
        if _get_rust_engine() is False:
            return False
        for pop in self.populations:
            if not _rust_supports_model(pop.model_name):
                return False
        return not any(p.plasticity for p in self.projections)

    def _rust_incompatibilities(self) -> list[str]:
        """List Python-only dynamics that the Rust backend would silently skip."""
        incompatible: list[str] = []
        if self._spike_gating:
            incompatible.append("spike_gating")
        if self.fim_lambda > 0:
            incompatible.append("fim_lambda")
        if self.state_monitors:
            incompatible.append("StateMonitor")
        if self.rate_monitors:
            incompatible.append("RateMonitor")
        return incompatible

    def _raise_for_rust_incompatibilities(self) -> None:
        """Fail fast when a caller forces Rust for Python-only semantics."""
        incompatible = self._rust_incompatibilities()
        if incompatible:
            names = ", ".join(incompatible)
            raise NotImplementedError(
                f"Rust backend does not support {names}; use backend='python' or backend='auto'"
            )

    def run(
        self,
        duration: float,
        dt: float = 0.001,
        progress: bool = False,
        backend: str = "auto",
        spike_gating: bool = False,
    ) -> None:
        """Run the simulation for *duration* seconds at timestep *dt*.

        *backend* selects execution: ``'auto'`` picks Rust when available
        and all models are supported, ``'rust'`` forces the Rust backend
        (raises if unavailable), ``'python'`` forces pure-Python,
        ``'mpi'`` runs MPI-distributed (requires mpi4py).

        *spike_gating*: skip neurons with zero input and near-rest voltage.
        Makes compute roughly proportional to active neuron count. Python
        backend only.

        The Rust backend preserves spike events and final population voltages.
        Per-step traces from ``StateMonitor`` and ``RateMonitor``, FIM
        feedback, and ``spike_gating`` remain Python-only; ``'auto'`` falls
        back to Python for those cases and forced ``'rust'`` raises.
        """
        self._spike_gating = spike_gating
        if backend == "mpi":
            return self._run_mpi(duration, dt)
        if backend == "rust":
            self._raise_for_rust_incompatibilities()
            return self._run_rust(duration, dt)
        if backend == "auto" and self._can_use_rust():
            return self._run_rust(duration, dt)
        return self._run_python(duration, dt, progress)

    def _run_mpi(self, duration: float, dt: float) -> None:
        # MPIRunner does not honour spike_gating or fim_lambda; refuse
        # rather than silently producing wrong results.
        if self._spike_gating:
            raise NotImplementedError(
                "spike_gating is not supported by the MPI backend; "
                "use backend='python' or rebuild without spike_gating"
            )
        if self.fim_lambda > 0:
            raise NotImplementedError(
                "fim_lambda > 0 (FIM feedback) is not supported by the MPI backend; "
                "use backend='python'"
            )
        if self.stimuli:
            raise NotImplementedError(
                "embedded stimuli are not supported by the MPI backend; "
                "use backend='python' or provide rank-local currents externally"
            )
        if self.state_monitors:
            raise NotImplementedError(
                "state monitors are not supported by the MPI backend; use backend='python'"
            )
        if any(proj.plasticity for proj in self.projections):
            raise NotImplementedError(
                "synaptic plasticity is not supported by the MPI backend; use backend='python'"
            )

        from .mpi_runner import MPIRunner

        n_steps = int(round(duration / dt))
        runner = MPIRunner(self)
        runner.run(n_steps, dt)

    def _run_rust(self, duration: float, dt: float) -> None:
        self._raise_for_rust_incompatibilities()
        engine_cls = _get_rust_engine()
        if engine_cls is False:
            raise RuntimeError("Rust engine not available")

        runner = engine_cls()
        pop_indices = {}
        for pop in self.populations:
            idx = runner.add_population(pop.model_name, pop.n)
            pop_indices[id(pop)] = idx

        for proj in self.projections:
            src_idx = pop_indices[id(proj.source)]
            tgt_idx = pop_indices[id(proj.target)]
            runner.add_projection(
                src_idx,
                tgt_idx,
                proj.indptr.tolist(),
                proj.indices.tolist(),
                proj.data.tolist(),
                proj.max_delay,
            )

        n_steps = int(round(duration / dt))
        results = runner.run(n_steps)

        for i, pop in enumerate(self.populations):
            # Sync voltages back from Rust
            if "voltages" in results and i < len(results["voltages"]):
                rust_v = results["voltages"][i]
                if len(rust_v) == pop.n:
                    pop.set_voltages(rust_v)

            # Decode spike events (u64: neuron_id << 32 | timestep)
            spike_arr = results["spike_data"][i]
            for mon in self.spike_monitors:
                if mon.population is pop:
                    for packed in spike_arr:
                        nid = int(packed >> 32)
                        t = int(packed & 0xFFFFFFFF)
                        mon.record_event(nid, t)

    def _run_python(self, duration: float, dt: float, progress: bool = False) -> None:
        self._rng = np.random.default_rng(self.seed)
        n_steps = int(round(duration / dt))

        pop_to_currents = {id(p): np.zeros(p.n, dtype=np.float64) for p in self.populations}
        last_spikes = {id(p): np.zeros(p.n, dtype=np.int8) for p in self.populations}
        report_interval = max(1, n_steps // 10) if progress else 0

        for t in range(n_steps):
            if report_interval and t % report_interval == 0:
                pct = int(100 * t / n_steps)
                sys.stdout.write(f"\r[{pct:3d}%] step {t}/{n_steps}")
                sys.stdout.flush()

            for pid in pop_to_currents:
                pop_to_currents[pid][:] = 0.0

            self._apply_stimuli(pop_to_currents, t, dt)
            self._apply_projections(pop_to_currents, last_spikes)

            for pop in self.populations:
                pid = id(pop)
                spikes = pop.step_all(pop_to_currents[pid], spike_gating=self._spike_gating)
                last_spikes[pid] = spikes
                self._record(pop, spikes, t, dt)

            self._update_plasticity(last_spikes)
            if self.fim_lambda > 0:
                self._apply_fim(last_spikes)

        if report_interval:
            sys.stdout.write(f"\r[100%] step {n_steps}/{n_steps}\n")
            sys.stdout.flush()

    def _apply_stimuli(
        self, pop_to_currents: dict[int, np.ndarray[Any, Any]], t: int, dt: float
    ) -> None:
        """Inject stimulus currents into target populations."""
        for stim in self.stimuli:
            target = stim.target
            if target is None:
                if self.populations:
                    target = self.populations[0]
                else:
                    continue
            pid = id(target)
            if pid not in pop_to_currents:
                continue
            if isinstance(stim, PoissonInput):
                pop_to_currents[pid][: stim.n] += stim.get_current(t, dt=dt)
            elif isinstance(stim, TimedArray):
                pop_to_currents[pid] += stim.get_current(t)
            elif isinstance(stim, StepCurrent):
                pop_to_currents[pid] += stim.get_current(t, dt)

    def _apply_projections(
        self,
        pop_to_currents: dict[int, np.ndarray[Any, Any]],
        last_spikes: dict[int, np.ndarray[Any, Any]],
    ) -> None:
        """Propagate spikes through all projections."""
        for proj in self.projections:
            src_spikes = last_spikes.get(id(proj.source), np.zeros(proj.source.n, dtype=np.int8))
            current = proj.propagate(src_spikes)
            pid = id(proj.target)
            if pid in pop_to_currents:
                pop_to_currents[pid] += current

    def _record(self, pop: Population, spikes: np.ndarray[Any, Any], t: int, dt: float) -> None:
        """Feed spikes/states to all monitors attached to this population."""
        for sp_mon in self.spike_monitors:
            if sp_mon.population is pop:
                sp_mon.record(spikes, t)
        for st_mon in self.state_monitors:
            if st_mon.population is pop:
                st_mon.snapshot(t)
        for rt_mon in self.rate_monitors:
            if rt_mon.population is pop:
                rt_mon.record(spikes, t, dt)

    def _update_plasticity(self, last_spikes: dict[int, np.ndarray[Any, Any]]) -> None:
        """Apply plasticity rules to projections that have them."""
        for proj in self.projections:
            if proj.plasticity:
                src_sp = last_spikes.get(id(proj.source), np.zeros(proj.source.n, dtype=np.int8))
                tgt_sp = last_spikes.get(id(proj.target), np.zeros(proj.target.n, dtype=np.int8))
                proj.update_plasticity(src_sp, tgt_sp)

    def _apply_fim(self, last_spikes: dict[int, np.ndarray[Any, Any]]) -> None:
        """Fisher Information Metric self-observation feedback.

        Pulls each neuron's activity toward the population mean,
        creating a global synchronisation force analogous to the FIM
        gradient in the Kuramoto model.

        ΔW_ij -= λ_fim · (activity_i - μ) / N

        Derived from scpn-quantum-control NB26-28: FIM alone synchronises
        (K=0, λ≥8), increases Φ by 73%, and is topology-universal.
        """
        lam = self.fim_lambda
        for proj in self.projections:
            src_sp = last_spikes.get(id(proj.source), np.zeros(proj.source.n))
            n_src = proj.source.n
            mu = float(np.mean(src_sp))
            deviation = src_sp.astype(np.float64) - mu
            for i in range(n_src):
                if deviation[i] == 0:
                    continue
                correction = lam * deviation[i] / n_src
                for k in range(proj.indptr[i], proj.indptr[i + 1]):
                    proj.data[k] = max(0.0, proj.data[k] - correction)

    def to_torch(
        self,
        surrogate_fn: Any | None = None,
    ) -> Any:
        """Build an explicit differentiable bridge without altering NumPy/Rust execution.

        The returned module accepts an input current tensor of shape
        ``(T, batch, input_dim)`` and runs the graph with the same
        previous-spike projection semantics used by the NumPy backend.
        """
        if self.stimuli:
            raise NotImplementedError(
                "Network.to_torch() does not support embedded stimuli; pass input currents "
                "through the returned module instead"
            )
        from ._torch_bridge import NetworkTorchBridge

        if surrogate_fn is None:
            from sc_neurocore.training.surrogate import atan_surrogate_custom_op

            surrogate_fn = atan_surrogate_custom_op

        return NetworkTorchBridge(
            self.populations,
            self.projections,
            surrogate_fn=surrogate_fn,
        )
