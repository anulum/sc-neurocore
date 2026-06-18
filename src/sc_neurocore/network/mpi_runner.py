# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MPI distributed simulation runner for billion-neuron scale

"""MPI-distributed network simulation.

Each MPI rank owns a subset of populations. Spikes are exchanged
via ``MPI_Allgatherv`` per timestep. Falls back gracefully when
``mpi4py`` is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

MPI: Any

try:
    from mpi4py import MPI as _MPI

    MPI = _MPI
    HAS_MPI = True
except ImportError:
    MPI = None
    HAS_MPI = False

from .population import Population
from .projection import Projection

if TYPE_CHECKING:
    from .network import Network


def _require_mpi() -> None:
    if not HAS_MPI:
        raise RuntimeError("mpi4py is required for MPI backend: pip install mpi4py")


def _get_rust_engine() -> Any:
    from .network import _get_rust_engine as get_network_rust_engine

    return get_network_rust_engine()


def _rust_supports_model(model_name: str) -> bool:
    from .network import _rust_supports_model as network_rust_supports_model

    return network_rust_supports_model(model_name)


class MPIRunner:
    """MPI-distributed network simulation.

    Partitions populations across MPI ranks via round-robin assignment.
    Each rank steps only its local populations; spikes propagate via
    ``MPI_Allgatherv`` every timestep.

    Each rank steps supported local populations through the Rust engine's
    ``step_population`` API when the extension is importable and every
    local model on the rank is supported. Otherwise the runner falls back
    to ``Population.step_all`` for CPU-only environments. ``spike_gating``
    and ``fim_lambda`` are unsupported by this runner — the
    ``Network._run_mpi`` dispatcher raises ``NotImplementedError`` when
    either is requested with ``backend='mpi'``.
    """

    def __init__(self, network: Network) -> None:
        _require_mpi()
        assert MPI is not None
        self.comm = MPI.COMM_WORLD
        self.rank: int = self.comm.Get_rank()
        self.size: int = self.comm.Get_size()
        self.network = network

        self._populations: list[Population] = network.populations
        self._projections: list[Projection] = network.projections
        self._local_indices: list[int] = []
        self._rank_of: dict[int, int] = {}
        self._partition_populations()

        self._cross_rank_projs: list[tuple[int, Projection]] = []
        self._local_projs: list[Projection] = []
        self._identify_cross_rank_projections()

        self._rust_runner: Any | None = None
        self._rust_local_pop_indices: dict[int, int] = {}
        self._rust_dispatch_enabled = False
        self._initialize_rust_dispatch()

    def _initialize_rust_dispatch(self) -> None:
        """Prepare a rank-local Rust runner when the installed engine supports it."""
        if not self._local_indices:
            return
        if not all(
            _rust_supports_model(self._populations[idx].model_name) for idx in self._local_indices
        ):
            return
        engine_cls = _get_rust_engine()
        if engine_cls is False:
            return
        runner = engine_cls()
        if not hasattr(runner, "step_population"):
            return
        for global_idx in self._local_indices:
            pop = self._populations[global_idx]
            rust_idx = runner.add_population(pop.model_name, pop.n)
            self._rust_local_pop_indices[global_idx] = int(rust_idx)
        self._rust_runner = runner
        self._rust_dispatch_enabled = True

    def _partition_populations(self) -> None:
        """Round-robin assignment of populations to ranks."""
        for i in range(len(self._populations)):
            owner = i % self.size
            self._rank_of[i] = owner
            if owner == self.rank:
                self._local_indices.append(i)

    def _identify_cross_rank_projections(self) -> None:
        """Separate projections into local and cross-rank."""
        pop_id_to_idx = {id(p): i for i, p in enumerate(self._populations)}
        for proj in self._projections:
            src_idx = pop_id_to_idx.get(id(proj.source), -1)
            tgt_idx = pop_id_to_idx.get(id(proj.target), -1)
            src_rank = self._rank_of.get(src_idx, -1)
            tgt_rank = self._rank_of.get(tgt_idx, -1)
            if src_rank != tgt_rank:
                self._cross_rank_projs.append((src_idx, proj))
            else:
                if tgt_rank == self.rank:
                    self._local_projs.append(proj)

    def _exchange_spikes(
        self, local_spikes: dict[int, np.ndarray[Any, Any]]
    ) -> dict[int, np.ndarray[Any, Any]]:
        """Allgatherv spike vectors so every rank knows who spiked.

        Each rank sends spike vectors for its local populations packed
        as (pop_index, n_neurons, spike_data...). Returns a dict of
        pop_index -> spike array for all populations.
        """
        assert MPI is not None
        chunks: list[np.ndarray[Any, Any]] = []
        for idx in self._local_indices:
            spikes = local_spikes.get(idx, np.zeros(self._populations[idx].n, dtype=np.int8))
            header = np.array([idx, spikes.shape[0]], dtype=np.int32)
            chunks.append(header.view(np.int8))
            chunks.append(spikes)

        send_buf = np.concatenate(chunks) if chunks else np.array([], dtype=np.int8)
        send_count = np.array(send_buf.shape[0], dtype=np.int32)
        recv_counts = np.empty(self.size, dtype=np.int32)
        self.comm.Allgather(send_count, recv_counts)

        total = int(recv_counts.sum())
        recv_buf = np.empty(total, dtype=np.int8)
        displacements = np.zeros(self.size, dtype=np.int32)
        for i in range(1, self.size):
            displacements[i] = displacements[i - 1] + recv_counts[i - 1]

        self.comm.Allgatherv(send_buf, [recv_buf, recv_counts, displacements, MPI.BYTE])

        all_spikes: dict[int, np.ndarray[Any, Any]] = {}
        pos = 0
        while pos < total:
            header = recv_buf[pos : pos + 8].view(np.int32)
            pop_idx = int(header[0])
            n = int(header[1])
            pos += 8
            all_spikes[pop_idx] = recv_buf[pos : pos + n].copy()
            pos += n

        return all_spikes

    def _step_local(
        self,
        pop_to_currents: dict[int, np.ndarray[Any, Any]],
        last_spikes: dict[int, np.ndarray[Any, Any]],
    ) -> dict[int, np.ndarray[Any, Any]]:
        """Step only this rank's populations, return local spike dict."""
        local_spikes: dict[int, np.ndarray[Any, Any]] = {}
        for idx in self._local_indices:
            pop = self._populations[idx]
            currents = np.asarray(
                pop_to_currents.get(idx, np.zeros(pop.n, dtype=np.float64)),
                dtype=np.float64,
            )
            if currents.shape != (pop.n,):
                raise ValueError(
                    f"current vector for population {idx} has shape {currents.shape}, "
                    f"expected {(pop.n,)}"
                )
            if self._rust_dispatch_enabled:
                assert self._rust_runner is not None
                result = self._rust_runner.step_population(
                    self._rust_local_pop_indices[idx],
                    np.ascontiguousarray(currents, dtype=np.float64),
                )
                spikes = np.asarray(result["spikes"], dtype=np.int8)
                voltages = np.asarray(result["voltages"], dtype=np.float64)
                if spikes.shape != (pop.n,):
                    raise RuntimeError(
                        f"Rust spike vector for population {idx} has shape {spikes.shape}, "
                        f"expected {(pop.n,)}"
                    )
                if voltages.shape != (pop.n,):
                    raise RuntimeError(
                        f"Rust voltage vector for population {idx} has shape {voltages.shape}, "
                        f"expected {(pop.n,)}"
                    )
                pop.set_voltages(voltages)
                spikes = spikes.copy()
            else:
                spikes = pop.step_all(currents)
            local_spikes[idx] = spikes
        return local_spikes

    def run(self, n_steps: int, dt: float = 0.001) -> None:
        """Run the distributed simulation for *n_steps* timesteps.

        Results are recorded via the network's monitors. Global monitors
        aggregate on rank 0 only.
        """
        np.random.seed(self.network.seed + self.rank)
        pop_id_to_idx = {id(p): i for i, p in enumerate(self._populations)}
        all_spikes: dict[int, np.ndarray[Any, Any]] = {
            i: np.zeros(p.n, dtype=np.int8) for i, p in enumerate(self._populations)
        }

        for t in range(n_steps):
            pop_to_currents: dict[int, np.ndarray[Any, Any]] = {
                idx: np.zeros(self._populations[idx].n, dtype=np.float64)
                for idx in self._local_indices
            }

            for proj in self._local_projs:
                src_idx = pop_id_to_idx[id(proj.source)]
                tgt_idx = pop_id_to_idx[id(proj.target)]
                src_sp = all_spikes.get(src_idx, np.zeros(proj.source.n, dtype=np.int8))
                current = proj.propagate(src_sp)
                if tgt_idx in pop_to_currents:
                    pop_to_currents[tgt_idx] += current

            for src_idx, proj in self._cross_rank_projs:
                tgt_idx = pop_id_to_idx[id(proj.target)]
                src_sp = all_spikes.get(src_idx, np.zeros(proj.source.n, dtype=np.int8))
                current = proj.propagate(src_sp)
                if tgt_idx in pop_to_currents:
                    pop_to_currents[tgt_idx] += current

            local_spikes = self._step_local(pop_to_currents, all_spikes)
            all_spikes = self._exchange_spikes(local_spikes)

            if self.rank == 0:
                net = self.network
                for mon in net.spike_monitors:
                    idx = pop_id_to_idx.get(id(mon.population))
                    if idx is not None and idx in all_spikes:
                        mon.record(all_spikes[idx], t)
                for mon in net.rate_monitors:  # type: ignore[assignment]
                    idx = pop_id_to_idx.get(id(mon.population))
                    if idx is not None and idx in all_spikes:
                        mon.record(all_spikes[idx], t, dt)  # type: ignore[call-arg]
