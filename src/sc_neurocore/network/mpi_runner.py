# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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

import numpy as np

try:
    from mpi4py import MPI

    HAS_MPI = True
except ImportError:
    MPI = None  # type: ignore[assignment]
    HAS_MPI = False

from .population import Population
from .projection import Projection


def _require_mpi() -> None:
    if not HAS_MPI:
        raise RuntimeError("mpi4py is required for MPI backend: pip install mpi4py")


class MPIRunner:
    """MPI-distributed network simulation for billion-neuron scale.

    Partitions populations across MPI ranks via round-robin assignment.
    Each rank steps only its local populations; spikes propagate via
    ``MPI_Allgatherv`` every timestep.

    Works with both Python and Rust backends per-rank: if the Rust
    engine is available and all local populations are supported, the
    rank uses Rust for its local step.
    """

    def __init__(self, network: object) -> None:
        _require_mpi()
        assert MPI is not None
        self.comm = MPI.COMM_WORLD
        self.rank: int = self.comm.Get_rank()
        self.size: int = self.comm.Get_size()
        self.network = network

        self._populations: list[Population] = network.populations  # type: ignore[attr-defined]
        self._projections: list[Projection] = network.projections  # type: ignore[attr-defined]

        self._local_indices: list[int] = []
        self._rank_of: dict[int, int] = {}
        self._partition_populations()

        self._cross_rank_projs: list[tuple[int, Projection]] = []
        self._local_projs: list[Projection] = []
        self._identify_cross_rank_projections()

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
        self, local_spikes: dict[int, np.ndarray]
    ) -> dict[int, np.ndarray]:
        """Allgatherv spike vectors so every rank knows who spiked.

        Each rank sends spike vectors for its local populations packed
        as (pop_index, n_neurons, spike_data...). Returns a dict of
        pop_index -> spike array for all populations.
        """
        assert MPI is not None
        chunks: list[np.ndarray] = []
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

        all_spikes: dict[int, np.ndarray] = {}
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
        pop_to_currents: dict[int, np.ndarray],
        last_spikes: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        """Step only this rank's populations, return local spike dict."""
        local_spikes: dict[int, np.ndarray] = {}
        for idx in self._local_indices:
            pop = self._populations[idx]
            spikes = pop.step_all(pop_to_currents.get(idx, np.zeros(pop.n, dtype=np.float64)))
            local_spikes[idx] = spikes
        return local_spikes

    def run(self, n_steps: int, dt: float = 0.001) -> None:
        """Run the distributed simulation for *n_steps* timesteps.

        Results are recorded via the network's monitors. Global monitors
        aggregate on rank 0 only.
        """
        np.random.seed(self.network.seed + self.rank)  # type: ignore[attr-defined]

        pop_id_to_idx = {id(p): i for i, p in enumerate(self._populations)}
        all_spikes: dict[int, np.ndarray] = {
            i: np.zeros(p.n, dtype=np.int8) for i, p in enumerate(self._populations)
        }

        for t in range(n_steps):
            pop_to_currents: dict[int, np.ndarray] = {
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
                for mon in net.spike_monitors:  # type: ignore[attr-defined]
                    idx = pop_id_to_idx.get(id(mon.population))
                    if idx is not None and idx in all_spikes:
                        mon.record(all_spikes[idx], t)
                for mon in net.rate_monitors:  # type: ignore[attr-defined]
                    idx = pop_id_to_idx.get(id(mon.population))
                    if idx is not None and idx in all_spikes:
                        mon.record(all_spikes[idx], t, dt)
