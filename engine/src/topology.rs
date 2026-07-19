// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust Ollivier-Ricci curvature (parity with src/sc_neurocore/math/topology.py)

//! Rust implementation of discrete Ollivier-Ricci curvature on a
//! coupling graph.
//!
//! Parity target: `ollivier_ricci_curvature` in
//! `src/sc_neurocore/math/topology.py`. For the same coupling matrix
//! and node pair, the Rust and Python paths return the same value to
//! within float64 round-off.
//!
//! References:
//!   Ollivier, Y. (2009). "Ricci curvature of Markov chains on metric
//!     spaces." J. Functional Analysis 256(3): 810-864.
//!
//! Definition:
//!   kappa(i, j) = 1 - W1(mu_i, mu_j) / d(i, j)
//! where d is the unweighted shortest-path (hop) distance, mu_i is the
//! lazy random walk distribution from node i (idleness 1/2, the rest
//! split by outgoing coupling weight), and W1 is the Wasserstein-1
//! (earth-mover) distance under the hop metric, solved exactly by a
//! successive-shortest-path min-cost flow.
//!
//! The min-cost-flow loop mirrors the Python reference iteration order
//! (Bellman-Ford with ascending node scan) so the chosen augmenting
//! paths — and therefore the floating-point accumulation of the
//! transport cost — are identical across the two implementations.

const IDLENESS: f64 = 0.5;
const TOLERANCE: f64 = 1e-12;

/// Outcome of an Ollivier-Ricci curvature request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CurvatureError {
    /// `knm` is not a square, non-empty matrix.
    BadShape,
    /// `knm` carries a non-finite or negative entry.
    BadValue,
    /// A node index is outside `[0, n)`.
    BadIndex,
    /// The transport sub-problem admitted no augmenting path.
    Infeasible,
}

/// Unweighted (hop-count) all-pairs shortest-path distances via BFS.
///
/// `graph` is row-major `n x n`. An edge exists where `graph[u*n+v] > 0`
/// (the diagonal is ignored). Unreachable pairs stay `f64::INFINITY`.
fn shortest_path_distances(graph: &[f64], n: usize) -> Vec<f64> {
    let mut distances = vec![f64::INFINITY; n * n];
    let mut queue: Vec<usize> = Vec::with_capacity(n);
    for source in 0..n {
        distances[source * n + source] = 0.0;
        queue.clear();
        queue.push(source);
        let mut head = 0;
        while head < queue.len() {
            let current = queue[head];
            head += 1;
            let next_distance = distances[source * n + current] + 1.0;
            for target in 0..n {
                if target == current || graph[current * n + target] <= 0.0 {
                    continue;
                }
                if next_distance < distances[source * n + target] {
                    distances[source * n + target] = next_distance;
                    queue.push(target);
                }
            }
        }
    }
    distances
}

/// Lazy random walk distribution from `node`: probability `IDLENESS`
/// stays, the remainder is split across outgoing edges by coupling
/// weight. A sink (no outgoing weight) keeps all mass at `node`.
fn lazy_random_walk(graph: &[f64], n: usize, node: usize) -> Vec<f64> {
    let mut distribution = vec![0.0; n];
    distribution[node] = IDLENESS;
    let mut row_sum = 0.0;
    for k in 0..n {
        if k != node {
            row_sum += graph[node * n + k];
        }
    }
    if row_sum == 0.0 {
        distribution[node] = 1.0;
        return distribution;
    }
    for k in 0..n {
        if k != node {
            distribution[k] += (1.0 - IDLENESS) * graph[node * n + k] / row_sum;
        }
    }
    distribution
}

/// Exact Wasserstein-1 cost between two distributions under the hop
/// metric `distances`, via a successive-shortest-path min-cost flow.
///
/// Returns `Ok(f64::INFINITY)` when some required transport edge has
/// infinite (unreachable) cost, matching the Python early return.
fn minimum_transport_cost(
    source: &[f64],
    target: &[f64],
    distances: &[f64],
    n: usize,
) -> Result<f64, CurvatureError> {
    let source_nodes: Vec<usize> = (0..n).filter(|&k| source[k] > 0.0).collect();
    let target_nodes: Vec<usize> = (0..n).filter(|&k| target[k] > 0.0).collect();
    if source_nodes.is_empty() || target_nodes.is_empty() {
        return Ok(0.0);
    }

    let total_supply = source_nodes.len();
    let total_demand = target_nodes.len();
    let mut costs = vec![0.0; total_supply * total_demand];
    for (s_idx, &s_node) in source_nodes.iter().enumerate() {
        for (d_idx, &d_node) in target_nodes.iter().enumerate() {
            let cost = distances[s_node * n + d_node];
            if !cost.is_finite() {
                return Ok(f64::INFINITY);
            }
            costs[s_idx * total_demand + d_idx] = cost;
        }
    }

    let source_id = total_supply + total_demand;
    let sink_id = source_id + 1;
    let node_count = sink_id + 1;
    let mut residual = vec![0.0; node_count * node_count];
    let mut edge_cost = vec![0.0; node_count * node_count];

    for (idx, &s_node) in source_nodes.iter().enumerate() {
        residual[source_id * node_count + idx] = source[s_node];
    }
    for (idx, &d_node) in target_nodes.iter().enumerate() {
        residual[(total_supply + idx) * node_count + sink_id] = target[d_node];
    }
    for s_idx in 0..total_supply {
        for d_idx in 0..total_demand {
            let u = s_idx;
            let v = total_supply + d_idx;
            let cost = costs[s_idx * total_demand + d_idx];
            residual[u * node_count + v] = f64::INFINITY;
            edge_cost[u * node_count + v] = cost;
            edge_cost[v * node_count + u] = -cost;
        }
    }

    let required: f64 = source.iter().sum();
    let mut transported = 0.0;
    let mut total_cost = 0.0;

    while transported + TOLERANCE < required {
        let mut dist = vec![f64::INFINITY; node_count];
        let mut parent = vec![usize::MAX; node_count];
        dist[source_id] = 0.0;
        for _ in 0..node_count - 1 {
            let mut updated = false;
            for u in 0..node_count {
                if !dist[u].is_finite() {
                    continue;
                }
                for v in 0..node_count {
                    if residual[u * node_count + v] <= TOLERANCE {
                        continue;
                    }
                    let candidate = dist[u] + edge_cost[u * node_count + v];
                    if candidate < dist[v] - TOLERANCE {
                        dist[v] = candidate;
                        parent[v] = u;
                        updated = true;
                    }
                }
            }
            if !updated {
                break;
            }
        }
        if parent[sink_id] == usize::MAX {
            return Err(CurvatureError::Infeasible);
        }

        let mut increment = required - transported;
        let mut v = sink_id;
        while v != source_id {
            let u = parent[v];
            increment = increment.min(residual[u * node_count + v]);
            v = u;
        }
        let mut v = sink_id;
        while v != source_id {
            let u = parent[v];
            residual[u * node_count + v] -= increment;
            residual[v * node_count + u] += increment;
            total_cost += increment * edge_cost[u * node_count + v];
            v = u;
        }
        transported += increment;
    }
    Ok(total_cost)
}

/// Discrete Ollivier-Ricci curvature between nodes `i` and `j`.
///
/// `knm` is row-major `n x n` and must be square, finite, and
/// non-negative (validated here so the PyO3 wrapper and the pure-Rust
/// callers share one contract). Returns `0.0` for `i == j` and for
/// node pairs at zero or infinite graph distance, matching the Python
/// reference.
pub fn ollivier_ricci_curvature(
    knm: &[f64],
    n: usize,
    i: usize,
    j: usize,
) -> Result<f64, CurvatureError> {
    if n == 0 || knm.len() != n * n {
        return Err(CurvatureError::BadShape);
    }
    for &value in knm {
        if !value.is_finite() || value < 0.0 {
            return Err(CurvatureError::BadValue);
        }
    }
    if i >= n || j >= n {
        return Err(CurvatureError::BadIndex);
    }
    if i == j {
        return Ok(0.0);
    }

    let distances = shortest_path_distances(knm, n);
    let graph_distance = distances[i * n + j];
    if !graph_distance.is_finite() || graph_distance <= 0.0 {
        return Ok(0.0);
    }
    let mu_i = lazy_random_walk(knm, n, i);
    let mu_j = lazy_random_walk(knm, n, j);
    let w1 = minimum_transport_cost(&mu_i, &mu_j, &distances, n)?;
    Ok(1.0 - w1 / graph_distance)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete_graph(n: usize) -> Vec<f64> {
        let mut g = vec![1.0; n * n];
        for k in 0..n {
            g[k * n + k] = 0.0;
        }
        g
    }

    #[test]
    fn self_pair_is_zero() {
        let g = complete_graph(4);
        assert_eq!(ollivier_ricci_curvature(&g, 4, 2, 2).unwrap(), 0.0);
    }

    #[test]
    fn complete_graph_is_positively_curved() {
        let g = complete_graph(5);
        let kappa = ollivier_ricci_curvature(&g, 5, 0, 1).unwrap();
        assert!(kappa > 0.0, "complete-graph curvature {kappa} not positive");
    }

    #[test]
    fn disconnected_pair_returns_zero() {
        // Two isolated edges: 0-1 and 2-3. Pair (0, 2) is disconnected.
        let mut g = vec![0.0; 16];
        g[1] = 1.0;
        g[4] = 1.0;
        g[11] = 1.0;
        g[14] = 1.0;
        let kappa = ollivier_ricci_curvature(&g, 4, 0, 2).unwrap();
        assert_eq!(kappa, 0.0);
    }

    #[test]
    fn ring_is_less_curved_than_complete() {
        // 6-cycle.
        let n = 6;
        let mut ring = vec![0.0; n * n];
        for k in 0..n {
            let a = k;
            let b = (k + 1) % n;
            ring[a * n + b] = 1.0;
            ring[b * n + a] = 1.0;
        }
        let kappa_ring = ollivier_ricci_curvature(&ring, n, 0, 1).unwrap();
        let complete = complete_graph(n);
        let kappa_complete = ollivier_ricci_curvature(&complete, n, 0, 1).unwrap();
        assert!(kappa_ring < kappa_complete);
    }

    #[test]
    fn rejects_bad_shape() {
        let g = vec![0.0; 6];
        assert_eq!(
            ollivier_ricci_curvature(&g, 3, 0, 1),
            Err(CurvatureError::BadShape)
        );
    }

    #[test]
    fn rejects_negative_entry() {
        let mut g = complete_graph(3);
        g[1] = -1.0;
        assert_eq!(
            ollivier_ricci_curvature(&g, 3, 0, 1),
            Err(CurvatureError::BadValue)
        );
    }

    #[test]
    fn rejects_out_of_range_index() {
        let g = complete_graph(3);
        assert_eq!(
            ollivier_ricci_curvature(&g, 3, 0, 5),
            Err(CurvatureError::BadIndex)
        );
    }
}
