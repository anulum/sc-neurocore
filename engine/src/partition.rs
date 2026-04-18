// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust KL refinement for HierarchicalPartitioner (parity with Python _refine)

//! Kernighan-Lin local refinement for the chiplet hierarchical
//! partitioner. Mirrors `HierarchicalPartitioner._refine` step-for-step
//! so the Rust output is bit-identical to the Python reference (the
//! algorithm is fully deterministic given a CSR adjacency, a list of
//! per-edge `|scc|` weights, vertex weights, an initial part_map and
//! `(kl_iterations, correlation_penalty, n_parts)`).
//!
//! Inputs (CSR-style flat arrays):
//!   - adj_offsets: [V+1] — row pointers; row i covers
//!     adj_neighbours[adj_offsets[i] .. adj_offsets[i+1]]
//!   - adj_neighbours: [E_total] — neighbour vertex ids
//!   - adj_scc_abs: [E_total] — `|edge_scc(v, n)|` for each adjacency
//!     entry (already-pre-computed so the kernel does NO graph lookup)
//!   - vertex_weights: [V] — `vw` per Python `graph.vertex_weights.get(v, 1.0)`
//!   - part_map: [V] (mut) — initial partition id per vertex; the
//!     kernel writes the refined assignment back into the same slot
//!
//! Behaviour: same as `_refine` — for each KL iteration, walk every
//! vertex of every partition, compute its full per-partition cost
//! vector in ONE neighbour scan, and move the vertex to the
//! best-gain target if the gain is strictly positive AND the source
//! partition has more than one vertex remaining. Stop early if no
//! moves happened in an iteration. Returns total moves performed.

pub fn kl_refine(
    adj_offsets: &[i64],
    adj_neighbours: &[i32],
    adj_scc_abs: &[f64],
    vertex_weights: &[f64],
    part_map: &mut [i32],
    parts_concat: &[i32],
    parts_offsets: &[i64],
    n_parts: i32,
    kl_iterations: i32,
    correlation_penalty: f64,
) -> u64 {
    let n_parts_us = n_parts as usize;

    // Seed per-partition vertex lists from the input concat array so
    // the iteration order EXACTLY mirrors Python's
    //     for i, part in enumerate(partitions):
    //         for v in list(part):    # snapshot of part NOW
    // — this ordering is load-bearing for the algorithm output (the
    // KL move order picks ties differently if vertex visit order
    // differs; the Python reference is the ground truth). Building
    // these from part_map alone (vertex-id order) gave 5/100
    // disagreement at V=100 — see commit notes.
    let mut parts: Vec<Vec<i32>> = vec![Vec::new(); n_parts_us];
    for i in 0..n_parts_us {
        let lo = parts_offsets[i] as usize;
        let hi = parts_offsets[i + 1] as usize;
        parts[i].extend_from_slice(&parts_concat[lo..hi]);
    }

    let mut weight_to: Vec<f64> = vec![0.0; n_parts_us];
    let mut total_moves: u64 = 0;

    for _ in 0..kl_iterations {
        let mut improved = false;
        for i in 0..n_parts_us {
            // Snapshot the part list at entry — matches Python
            // `for v in list(part)`. Subsequent moves into THIS
            // part within this iteration won't be revisited until
            // the next outer i sees them.
            let snapshot: Vec<i32> = parts[i].clone();
            for v_i32 in snapshot {
                let v = v_i32 as usize;
                if parts[i].len() <= 1 {
                    continue;
                }
                let cur_part = part_map[v];
                if (cur_part as usize) != i {
                    // v was moved out earlier in this same iter — skip
                    // (Python's `for v in list(part)` would still visit
                    // it, but `part.remove(v)` would have raised since
                    // v is no longer in the list. Defensive guard.)
                    continue;
                }
                let vw = vertex_weights[v];

                // Single-pass per-partition cost vector.
                for w in weight_to.iter_mut() {
                    *w = 0.0;
                }
                let mut total_weight: f64 = 0.0;
                let lo = adj_offsets[v] as usize;
                let hi = adj_offsets[v + 1] as usize;
                for k in lo..hi {
                    let n = adj_neighbours[k] as usize;
                    let scc = adj_scc_abs[k];
                    let contrib = vw * (1.0 + scc * correlation_penalty);
                    total_weight += contrib;
                    let tgt = part_map[n];
                    if (0..n_parts).contains(&tgt) {
                        weight_to[tgt as usize] += contrib;
                    }
                }

                let current_cost = total_weight - weight_to[i];
                let mut best_target = i as i32;
                let mut best_gain: f64 = 0.0;
                for j in 0..n_parts_us {
                    if j == i {
                        continue;
                    }
                    let cand_cost = total_weight - weight_to[j];
                    let gain = current_cost - cand_cost;
                    if gain > best_gain {
                        best_gain = gain;
                        best_target = j as i32;
                    }
                }

                if best_target != (i as i32) && best_gain > 0.0 {
                    // Remove v from parts[i] (linear scan, O(|part_i|)
                    // — same complexity as Python's list.remove).
                    let pos = parts[i].iter().position(|&x| x == v_i32).expect("v in parts[i]");
                    parts[i].remove(pos);
                    parts[best_target as usize].push(v_i32);
                    part_map[v] = best_target;
                    total_moves += 1;
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }

    total_moves
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph_zero_moves() {
        let part: &mut [i32] = &mut [];
        // n_parts=2 → parts_offsets needs P+1=3 entries
        let n = kl_refine(&[0], &[], &[], &[], part, &[], &[0, 0, 0], 2, 5, 0.5);
        assert_eq!(n, 0);
    }

    #[test]
    fn single_partition_no_moves() {
        let mut pm = vec![0i32, 0, 0, 0];
        let offsets = vec![0i64, 1, 2, 3, 4];
        let neighbours = vec![1i32, 0, 3, 2];
        let scc = vec![0.1, 0.1, 0.1, 0.1];
        let vw = vec![1.0; 4];
        let parts_concat = vec![0i32, 1, 2, 3];
        let parts_offsets = vec![0i64, 4];
        let n = kl_refine(&offsets, &neighbours, &scc, &vw, &mut pm,
                          &parts_concat, &parts_offsets, 1, 5, 0.5);
        assert_eq!(n, 0);
        assert_eq!(pm, vec![0, 0, 0, 0]);
    }

    #[test]
    fn two_partitions_isolated_swap() {
        let offsets = vec![0i64, 1, 2, 3, 4];
        let neighbours = vec![1i32, 0, 3, 2];
        let scc = vec![0.5, 0.5, 0.5, 0.5];
        let vw = vec![1.0; 4];
        let mut pm = vec![0i32, 1, 0, 1];
        let parts_concat = vec![0i32, 2, 1, 3];
        let parts_offsets = vec![0i64, 2, 4];
        let n = kl_refine(&offsets, &neighbours, &scc, &vw, &mut pm,
                          &parts_concat, &parts_offsets, 2, 5, 0.5);
        assert!(n > 0);
    }

    #[test]
    fn part_size_one_vertex_pinned() {
        let offsets = vec![0i64, 2, 3, 4];
        let neighbours = vec![1i32, 2, 0, 0];
        let scc = vec![0.1, 0.1, 0.1, 0.1];
        let vw = vec![1.0; 3];
        let mut pm = vec![0i32, 1, 1];
        let parts_concat = vec![0i32, 1, 2];
        let parts_offsets = vec![0i64, 1, 3];
        let _ = kl_refine(&offsets, &neighbours, &scc, &vw, &mut pm,
                          &parts_concat, &parts_offsets, 2, 5, 0.5);
        assert_eq!(pm[0], 0, "lone-vertex partition must stay populated");
    }
}
