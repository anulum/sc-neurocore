// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety acceleration for topology

use std::collections::VecDeque;
use std::f64::consts::PI;

fn validate_graph(knm: &[Vec<f64>]) -> Result<usize, &'static str> {
    let n = knm.len();
    if n == 0 {
        return Err("knm must contain at least one node");
    }
    for row in knm {
        if row.len() != n {
            return Err("knm must be a square coupling matrix");
        }
        for value in row {
            if !value.is_finite() {
                return Err("knm must contain only finite values");
            }
            if *value < 0.0 {
                return Err("knm must be non-negative for Ollivier-Ricci curvature");
            }
        }
    }
    Ok(n)
}

fn shortest_path_distances(knm: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = knm.len();
    let mut distances = vec![vec![f64::INFINITY; n]; n];
    for source in 0..n {
        distances[source][source] = 0.0;
        let mut queue = VecDeque::from([source]);
        while let Some(current) = queue.pop_front() {
            let next_distance = distances[source][current] + 1.0;
            for target in 0..n {
                if target == current || knm[current][target] <= 0.0 {
                    continue;
                }
                if next_distance < distances[source][target] {
                    distances[source][target] = next_distance;
                    queue.push_back(target);
                }
            }
        }
    }
    distances
}

fn lazy_random_walk(knm: &[Vec<f64>], node: usize, idleness: f64) -> Vec<f64> {
    let n = knm.len();
    let mut distribution = vec![0.0; n];
    distribution[node] = idleness;
    let row_sum: f64 = knm[node]
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx != node)
        .map(|(_, value)| *value)
        .sum();
    if row_sum == 0.0 {
        distribution[node] = 1.0;
        return distribution;
    }
    for (target, value) in knm[node].iter().enumerate() {
        if target != node {
            distribution[target] += (1.0 - idleness) * value / row_sum;
        }
    }
    distribution
}

fn minimum_transport_cost(source: &[f64], target: &[f64], distances: &[Vec<f64>]) -> Result<f64, &'static str> {
    let source_nodes: Vec<usize> = source
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| if *value > 0.0 { Some(idx) } else { None })
        .collect();
    let target_nodes: Vec<usize> = target
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| if *value > 0.0 { Some(idx) } else { None })
        .collect();
    if source_nodes.is_empty() || target_nodes.is_empty() {
        return Ok(0.0);
    }
    let supply_count = source_nodes.len();
    let demand_count = target_nodes.len();
    let source_id = supply_count + demand_count;
    let sink_id = source_id + 1;
    let node_count = sink_id + 1;
    let mut residual = vec![vec![0.0; node_count]; node_count];
    let mut edge_cost = vec![vec![0.0; node_count]; node_count];
    for (idx, node) in source_nodes.iter().enumerate() {
        residual[source_id][idx] = source[*node];
    }
    for (idx, node) in target_nodes.iter().enumerate() {
        residual[supply_count + idx][sink_id] = target[*node];
    }
    for (s_idx, source_node) in source_nodes.iter().enumerate() {
        for (d_idx, target_node) in target_nodes.iter().enumerate() {
            let cost = distances[*source_node][*target_node];
            if !cost.is_finite() {
                return Ok(f64::INFINITY);
            }
            let u = s_idx;
            let v = supply_count + d_idx;
            residual[u][v] = f64::INFINITY;
            edge_cost[u][v] = cost;
            edge_cost[v][u] = -cost;
        }
    }
    let required: f64 = source.iter().sum();
    let mut transported = 0.0;
    let mut total_cost = 0.0;
    let tolerance = 1e-12;
    while transported + tolerance < required {
        let mut dist = vec![f64::INFINITY; node_count];
        let mut parent = vec![usize::MAX; node_count];
        dist[source_id] = 0.0;
        for _ in 0..node_count.saturating_sub(1) {
            let mut updated = false;
            for u in 0..node_count {
                if !dist[u].is_finite() {
                    continue;
                }
                for v in 0..node_count {
                    if residual[u][v] <= tolerance {
                        continue;
                    }
                    let candidate = dist[u] + edge_cost[u][v];
                    if candidate < dist[v] - tolerance {
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
            return Err("transport problem is infeasible");
        }
        let mut increment = required - transported;
        let mut v = sink_id;
        while v != source_id {
            let u = parent[v];
            increment = increment.min(residual[u][v]);
            v = u;
        }
        v = sink_id;
        while v != source_id {
            let u = parent[v];
            residual[u][v] -= increment;
            residual[v][u] += increment;
            total_cost += increment * edge_cost[u][v];
            v = u;
        }
        transported += increment;
    }
    Ok(total_cost)
}

pub fn winding_number(phases: &[f64]) -> Result<i64, &'static str> {
    if phases.iter().any(|value| !value.is_finite()) {
        return Err("phases must be finite");
    }
    let mut total = 0.0;
    for pair in phases.windows(2) {
        let mut diff = pair[1] - pair[0];
        if diff > PI {
            diff -= 2.0 * PI;
        }
        if diff < -PI {
            diff += 2.0 * PI;
        }
        total += diff;
    }
    Ok((total / (2.0 * PI)).round() as i64)
}

pub fn ollivier_ricci_curvature(knm: &[Vec<f64>], i: usize, j: usize) -> Result<f64, &'static str> {
    let n = validate_graph(knm)?;
    if i >= n || j >= n {
        return Err("node index out of range for coupling graph");
    }
    if i == j {
        return Ok(0.0);
    }
    let distances = shortest_path_distances(knm);
    let graph_distance = distances[i][j];
    if !graph_distance.is_finite() || graph_distance <= 0.0 {
        return Ok(0.0);
    }
    let mu_i = lazy_random_walk(knm, i, 0.5);
    let mu_j = lazy_random_walk(knm, j, 0.5);
    let w1 = minimum_transport_cost(&mu_i, &mu_j, &distances)?;
    if !w1.is_finite() {
        return Ok(0.0);
    }
    Ok(1.0 - w1 / graph_distance)
}

pub fn sheaf_consistency_defect(phases: &[f64], knm: &[Vec<f64>]) -> Result<f64, &'static str> {
    let n = validate_graph(knm)?;
    if phases.len() != n || phases.iter().any(|value| !value.is_finite()) {
        return Err("phases must be finite and match knm size");
    }
    let mut cost = 0.0;
    for i in 0..n {
        for j in 0..n {
            cost += knm[i][j].abs() * (1.0 - (phases[j] - phases[i]).cos());
        }
    }
    Ok(cost / ((n * n) as f64))
}

pub fn connection_curvature(phases: &[f64], knm: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, &'static str> {
    let n = validate_graph(knm)?;
    if phases.len() != n || phases.iter().any(|value| !value.is_finite()) {
        return Err("phases must be finite and match knm size");
    }
    let mut curvature = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            curvature[i][j] = knm[i][j] * (phases[j] - phases[i]).cos();
        }
    }
    Ok(curvature)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_graph_lazy_ricci_matches_python_contract() {
        let mut graph = vec![vec![1.0; 4]; 4];
        for (idx, row) in graph.iter_mut().enumerate() {
            row[idx] = 0.0;
        }
        let kappa = ollivier_ricci_curvature(&graph, 0, 1).unwrap();
        assert!((kappa - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_invalid_graph_contract() {
        let graph = vec![vec![0.0, -1.0], vec![1.0, 0.0]];
        assert!(ollivier_ricci_curvature(&graph, 0, 1).is_err());
    }
}
