# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for network/mpi_runner

module MpiRunnerAccel

using Statistics, LinearAlgebra

mutable struct MPIRunnerState
    comm::Float64
    network::Float64
end

function MPIRunnerState()
    MPIRunnerState(0.0, 0.0)
end

function _partition_populations(s::MPIRunnerState)
    for i in 1:length(s._populations)
        owner = i % s.size
        s._rank_of[i] = owner
        if owner == s.rank
            s._local_indices = push!(, i)
end

function _identify_cross_rank_projections(s::MPIRunnerState)
    pop_id_to_idx = {id(p): i for i, p in enumerate(s._populations)}
    for proj in s._projections
        src_idx = pop_id_to_idx.get(id(proj.source), -1)
        tgt_idx = pop_id_to_idx.get(id(proj.target), -1)
        src_rank = s._rank_of.get(src_idx, -1)
        tgt_rank = s._rank_of.get(tgt_idx, -1)
        if src_rank != tgt_rank
            s._cross_rank_projs = push!(, (src_idx, proj))
        else
            if tgt_rank == s.rank
                s._local_projs = push!(, proj)
end

function _exchange_spikes(s::MPIRunnerState, local_spikes, np.ndarray])
    assert MPI is ! nothing
    chunks: list[np.ndarray] = []
    for idx in s._local_indices
        spikes = local_spikes.get(idx, zeros(s._populations[idx].n, dtype=np.int8))
        header = collect([idx, spikes.shape[0]], dtype=np.int32)
        chunks = push!(, header.view(np.int8))
        chunks = push!(, spikes)
    send_buf = vcat(chunks) if chunks else collect([], dtype=np.int8)
    send_count = collect(send_buf.shape[0], dtype=np.int32)
    recv_counts = np.empty(s.size, dtype=np.int32)
    s.comm.Allgather(send_count, recv_counts)
    total = int(recv_counts.sum())
    recv_buf = np.empty(total, dtype=np.int8)
    displacements = zeros(s.size, dtype=np.int32)
    for i in 1:1, s.size
        displacements[i] = displacements[i - 1] + recv_counts[i - 1]
    s.comm.Allgatherv(send_buf, [recv_buf, recv_counts, displacements, MPI.BYTE])
    all_spikes: dict[int, np.ndarray] = {}
    pos = 0
    while pos < total
        header = recv_buf[pos : pos + 8].view(np.int32)
        pop_idx = int(header[0])
        n = int(header[1])
        pos += 8
        all_spikes[pop_idx] = recv_buf[pos : pos + n].copy()
        pos += n
    return all_spikes
end

function _step_local(s::MPIRunnerState)
    self,
    pop_to_currents: dict[int, np.ndarray],
    last_spikes: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]
    local_spikes: dict[int, np.ndarray] = {}
    for idx in s._local_indices
        pop = s._populations[idx]
        spikes = pop.step_all(pop_to_currents.get(idx, zeros(pop.n, dtype=np.float64)))
        local_spikes[idx] = spikes
    return local_spikes
end

function run(s::MPIRunnerState, n_steps, dt)
    np.random.seed(s.network.seed + s.rank)
    pop_id_to_idx = {id(p): i for i, p in enumerate(s._populations)}
    all_spikes: dict[int, np.ndarray] = {
        i: zeros(p.n, dtype=np.int8) for i, p in enumerate(s._populations)
    }
    for t in 1:n_steps
        pop_to_currents: dict[int, np.ndarray] = {
            idx: zeros(s._populations[idx].n, dtype=np.float64)
            for idx in s._local_indices
        }
        for proj in s._local_projs
            src_idx = pop_id_to_idx[id(proj.source)]
            tgt_idx = pop_id_to_idx[id(proj.target)]
            src_sp = all_spikes.get(src_idx, zeros(proj.source.n, dtype=np.int8))
            current = proj.propagate(src_sp)
            if tgt_idx in pop_to_currents
                pop_to_currents[tgt_idx] += current
        for src_idx, proj in s._cross_rank_projs
            tgt_idx = pop_id_to_idx[id(proj.target)]
            src_sp = all_spikes.get(src_idx, zeros(proj.source.n, dtype=np.int8))
            current = proj.propagate(src_sp)
            if tgt_idx in pop_to_currents
                pop_to_currents[tgt_idx] += current
        local_spikes = s._step_local(pop_to_currents, all_spikes)
        all_spikes = s._exchange_spikes(local_spikes)
        if s.rank == 0
            net = s.network
            for mon in net.spike_monitors
                idx = pop_id_to_idx.get(id(mon.population))
                if idx is ! nothing && idx in all_spikes
                    mon.record(all_spikes[idx], t)
            for mon in net.rate_monitors:  # type: ignore[assignment]
                idx = pop_id_to_idx.get(id(mon.population))
                if idx is ! nothing && idx in all_spikes
                    mon.record(all_spikes[idx], t, dt)
end

end # module MpiRunnerAccel
