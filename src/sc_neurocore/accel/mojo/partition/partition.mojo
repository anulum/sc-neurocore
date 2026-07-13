# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo KL refine for HierarchicalPartitioner (parity with Rust)
#
# Build:
#   ~/.pixi/bin/mojo build --emit shared-lib -o libpartition.so partition.mojo
#
# Per `feedback_mojo_026_ffi_pattern`: the @export decorator forbids
# parametric signatures, so every input/output buffer is passed as a
# raw `Int` address (numpy `arr.ctypes.data`) and the Mojo body
# reconstructs `UnsafePointer[T, MutAnyOrigin](unsafe_from_address=...)`
# inside the function.
#
# Iteration order MUST mirror Python + Rust + Julia + Go to keep the
# part_map output bit-exact: per-partition vertex list, snapshot at
# outer-i entry, linear-scan remove + append on move.

from std.memory import UnsafePointer, alloc


@always_inline
def _ptr_i64(addr: Int) -> UnsafePointer[Int64, MutAnyOrigin]:
    return UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=addr)


@always_inline
def _ptr_i32(addr: Int) -> UnsafePointer[Int32, MutAnyOrigin]:
    return UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=addr)


@always_inline
def _ptr_f64(addr: Int) -> UnsafePointer[Float64, MutAnyOrigin]:
    return UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=addr)


@export
def kl_refine_c(
    adj_offsets_addr: Int,
    adj_neighbours_addr: Int,
    adj_scc_abs_addr: Int,
    vertex_weights_addr: Int,
    part_map_addr: Int,
    parts_concat_addr: Int,
    parts_offsets_addr: Int,
    v_total: Int,
    e_total: Int,
    n_parts: Int32,
    kl_iterations: Int32,
    correlation_penalty: Float64,
) -> UInt64:
    var adj_offsets = _ptr_i64(adj_offsets_addr)
    var adj_neighbours = _ptr_i32(adj_neighbours_addr)
    var adj_scc_abs = _ptr_f64(adj_scc_abs_addr)
    var vertex_weights = _ptr_f64(vertex_weights_addr)
    var part_map = _ptr_i32(part_map_addr)
    var parts_concat = _ptr_i32(parts_concat_addr)
    var parts_offsets = _ptr_i64(parts_offsets_addr)

    var n_parts_i = Int(n_parts)
    var V = v_total

    # Seed per-partition vertex lists from input concat to preserve
    # Python's `for v in list(part)` iteration order.
    var raw_buf = alloc[Int32](V * n_parts_i)
    var parts_buf = UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(raw_buf))
    var raw_len = alloc[Int64](n_parts_i)
    var parts_len = UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=Int(raw_len))
    var raw_snap = alloc[Int32](V)
    var snapshot = UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(raw_snap))
    var raw_wt = alloc[Float64](n_parts_i)
    var weight_to = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=Int(raw_wt))

    for p in range(n_parts_i):
        var lo = Int(parts_offsets[p])
        var hi = Int(parts_offsets[p + 1])
        parts_len[p] = Int64(hi - lo)
        for k in range(lo, hi):
            parts_buf[p * V + (k - lo)] = parts_concat[k]

    var total_moves: UInt64 = 0
    var cp = correlation_penalty
    var kl = Int(kl_iterations)

    for _ in range(kl):
        var improved = False
        for i in range(n_parts_i):
            # Snapshot the part list at entry to mirror Python's
            # `for v in list(part)`.
            var snap_n = Int(parts_len[i])
            for k in range(snap_n):
                snapshot[k] = parts_buf[i * V + k]
            for s in range(snap_n):
                var v32 = snapshot[s]
                var v = Int(v32)
                if parts_len[i] <= 1:
                    continue
                if Int(part_map[v]) != i:
                    continue
                var vw = vertex_weights[v]

                for q in range(n_parts_i):
                    weight_to[q] = 0.0
                var total_weight: Float64 = 0.0
                var lo = Int(adj_offsets[v])
                var hi = Int(adj_offsets[v + 1])
                for k_idx in range(lo, hi):
                    var n = Int(adj_neighbours[k_idx])
                    var contrib = vw * (1.0 + adj_scc_abs[k_idx] * cp)
                    total_weight += contrib
                    var tgt = Int(part_map[n])
                    if 0 <= tgt and tgt < n_parts_i:
                        weight_to[tgt] += contrib

                var current_cost = total_weight - weight_to[i]
                var best_target = Int32(i)
                var best_gain: Float64 = 0.0
                for j in range(n_parts_i):
                    if j == i:
                        continue
                    var gain = current_cost - (total_weight - weight_to[j])
                    if gain > best_gain:
                        best_gain = gain
                        best_target = Int32(j)

                if Int(best_target) != i and best_gain > 0.0:
                    # Linear scan remove from parts_buf[i, :len_i].
                    var len_i = Int(parts_len[i])
                    var pos = -1
                    for k_pos in range(len_i):
                        if parts_buf[i * V + k_pos] == v32:
                            pos = k_pos
                            break
                    if pos >= 0:
                        for k_shift in range(pos, len_i - 1):
                            parts_buf[i * V + k_shift] = parts_buf[i * V + k_shift + 1]
                        parts_len[i] = Int64(len_i - 1)
                    var bt = Int(best_target)
                    var bt_len = Int(parts_len[bt])
                    parts_buf[bt * V + bt_len] = v32
                    parts_len[bt] = Int64(bt_len + 1)
                    part_map[v] = best_target
                    total_moves += 1
                    improved = True
        if not improved:
            break

    var raw_buf_free = UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=Int(parts_buf))
    raw_buf_free.free()
    var raw_len_free = UnsafePointer[Int64, MutExternalOrigin](unsafe_from_address=Int(parts_len))
    raw_len_free.free()
    var raw_snap_free = UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=Int(snapshot))
    raw_snap_free.free()
    var raw_wt_free = UnsafePointer[Float64, MutExternalOrigin](unsafe_from_address=Int(weight_to))
    raw_wt_free.free()

    return total_moves
