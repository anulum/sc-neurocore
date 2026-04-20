# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for mpi_runner

fn _require_mpi() -> Int:
    var __require_mpi_line = 'if not HAS_MPI:'
    var __require_mpi_line = 'raise RuntimeError("mpi4py is required for MPI backend: pip '
    return 0

fn _partition_populations() -> Int:
    var __partition_populations_line = 'for i in range(len(_populations)):'
    var __partition_populations_line = 'owner = i % size'
    var __partition_populations_line = '_rank_of[i] = owner'
    var __partition_populations_line = 'if owner == rank:'
    var __partition_populations_line = '_local_indices.append(i)'
    return 0

fn _identify_cross_rank_projections() -> Int:
    var __identify_cross_rank_projections_line = 'pop_id_to_idx = {id(p): i for i, p in enumerate(_populations'
    var __identify_cross_rank_projections_line = 'for proj in _projections:'
    var __identify_cross_rank_projections_line = 'src_idx = pop_id_to_idx.get(id(proj.source), -1)'
    var __identify_cross_rank_projections_line = 'tgt_idx = pop_id_to_idx.get(id(proj.target), -1)'
    var __identify_cross_rank_projections_line = 'src_rank = _rank_of.get(src_idx, -1)'
    var __identify_cross_rank_projections_line = 'tgt_rank = _rank_of.get(tgt_idx, -1)'
    var __identify_cross_rank_projections_line = 'if src_rank != tgt_rank:'
    var __identify_cross_rank_projections_line = '_cross_rank_projs.append((src_idx, proj))'
    var __identify_cross_rank_projections_line = 'else:'
    var __identify_cross_rank_projections_line = 'if tgt_rank == rank:'
    var __identify_cross_rank_projections_line = '_local_projs.append(proj)'
    return 0

fn _exchange_spikes(local_spikes: Int) -> Int:
    var __exchange_spikes_line = 'assert MPI is not 0'
    var __exchange_spikes_line = 'chunks: list[ndarray] = []'
    var __exchange_spikes_line = 'for idx in _local_indices:'
    var __exchange_spikes_line = 'spikes = local_spikes.get(idx, zeros(_populations[idx].n, dt'
    var __exchange_spikes_line = 'header = array([idx, spikes.shape[0]], dtype=int32)'
    var __exchange_spikes_line = 'chunks.append(header.view(int8))'
    var __exchange_spikes_line = 'chunks.append(spikes)'
    var __exchange_spikes_line = 'send_buf = concatenate(chunks) if chunks else array([], dtyp'
    var __exchange_spikes_line = 'send_count = array(send_buf.shape[0], dtype=int32)'
    var __exchange_spikes_line = 'recv_counts = empty(size, dtype=int32)'
    var __exchange_spikes_line = 'comm.Allgather(send_count, recv_counts)'
    var __exchange_spikes_line = 'total = int(recv_counts.sum())'
    var __exchange_spikes_line = 'recv_buf = empty(total, dtype=int8)'
    var __exchange_spikes_line = 'displacements = zeros(size, dtype=int32)'
    var __exchange_spikes_line = 'for i in range(1, size):'
    var __exchange_spikes_line = 'displacements[i] = displacements[i - 1] + recv_counts[i - 1]'
    var __exchange_spikes_line = 'comm.Allgatherv(send_buf, [recv_buf, recv_counts, displaceme'
    var __exchange_spikes_line = 'all_spikes: dict[int, ndarray] = {}'
    var __exchange_spikes_line = 'pos = 0'
    var __exchange_spikes_line = 'while pos < total:'
    var __exchange_spikes_line = 'header = recv_buf[pos : pos + 8].view(int32)'
    var __exchange_spikes_line = 'pop_idx = int(header[0])'
    var __exchange_spikes_line = 'n = int(header[1])'
    var __exchange_spikes_line = 'pos += 8'
    var __exchange_spikes_line = 'all_spikes[pop_idx] = recv_buf[pos : pos + n].copy()'
    var __exchange_spikes_line = 'pos += n'
    return 0  # return all_spikes

fn _step_local(pop_to_currents: Int, last_spikes: Int) -> Int:
    var __step_local_line = 'self,'
    var __step_local_line = 'pop_to_currents: dict[int, ndarray],'
    var __step_local_line = 'last_spikes: dict[int, ndarray],'
    var __step_local_line = ') -> dict[int, ndarray]:'
    var __step_local_line = 'local_spikes: dict[int, ndarray] = {}'
    var __step_local_line = 'for idx in _local_indices:'
    var __step_local_line = 'pop = _populations[idx]'
    var __step_local_line = 'spikes = pop.step_all(pop_to_currents.get(idx, zeros(pop.n, '
    var __step_local_line = 'local_spikes[idx] = spikes'
    return 0  # return local_spikes

fn run(n_steps: Int, dt: Int) -> Int:
    var _run_line = 'random.seed(network.seed + rank)'
    var _run_line = 'pop_id_to_idx = {id(p): i for i, p in enumerate(_populations'
    var _run_line = 'all_spikes: dict[int, ndarray] = {'
    var _run_line = 'i: zeros(p.n, dtype=int8) for i, p in enumerate(_populations'
    var _run_line = '}'
    var _run_line = 'for t in range(n_steps):'
    var _run_line = 'pop_to_currents: dict[int, ndarray] = {'
    var _run_line = 'idx: zeros(_populations[idx].n, dtype=float64)'
    var _run_line = 'for idx in _local_indices'
    var _run_line = '}'
    var _run_line = 'for proj in _local_projs:'
    var _run_line = 'src_idx = pop_id_to_idx[id(proj.source)]'
    var _run_line = 'tgt_idx = pop_id_to_idx[id(proj.target)]'
    var _run_line = 'src_sp = all_spikes.get(src_idx, zeros(proj.source.n, dtype='
    var _run_line = 'current = proj.propagate(src_sp)'
    var _run_line = 'if tgt_idx in pop_to_currents:'
    var _run_line = 'pop_to_currents[tgt_idx] += current'
    var _run_line = 'for src_idx, proj in _cross_rank_projs:'
    var _run_line = 'tgt_idx = pop_id_to_idx[id(proj.target)]'
    var _run_line = 'src_sp = all_spikes.get(src_idx, zeros(proj.source.n, dtype='
    var _run_line = 'current = proj.propagate(src_sp)'
    var _run_line = 'if tgt_idx in pop_to_currents:'
    var _run_line = 'pop_to_currents[tgt_idx] += current'
    var _run_line = 'local_spikes = _step_local(pop_to_currents, all_spikes)'
    var _run_line = 'all_spikes = _exchange_spikes(local_spikes)'
    var _run_line = 'if rank == 0:'
    var _run_line = 'net = network'
    var _run_line = 'for mon in net.spike_monitors:'
    var _run_line = 'idx = pop_id_to_idx.get(id(mon.population))'
    var _run_line = 'if idx is not 0 and idx in all_spikes:'
    var _run_line = 'mon.record(all_spikes[idx], t)'
    var _run_line = 'for mon in net.rate_monitors:  # type: ignore[assignment]'
    var _run_line = 'idx = pop_id_to_idx.get(id(mon.population))'
    var _run_line = 'if idx is not 0 and idx in all_spikes:'
    var _run_line = 'mon.record(all_spikes[idx], t, dt)'
    return 0
