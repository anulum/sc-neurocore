// SPDX-License-Identifier: AGPL-3.0-or-later
//! Biologically plausible connectivity generators.

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Watts-Strogatz small-world network.
///
/// Returns adjacency matrix as flat row-major Vec<u8> of shape [n, n].
pub fn watts_strogatz(n: usize, k: usize, p_rewire: f64, seed: u64) -> Vec<u8> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut adj = vec![0u8; n * n];

    if k >= n {
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    adj[i * n + j] = 1;
                }
            }
        }
        return adj;
    }

    // Ring lattice
    for i in 0..n {
        for d in 1..=k / 2 {
            let j = (i + d) % n;
            adj[i * n + j] = 1;
            adj[j * n + i] = 1;
        }
    }

    // Rewire
    for i in 0..n {
        for d in 1..=k / 2 {
            let j = (i + d) % n;
            if rng.random::<f64>() < p_rewire {
                adj[i * n + j] = 0;
                let mut new_j = i;
                while new_j == i || adj[i * n + new_j] == 1 {
                    new_j = (rng.random::<f64>() * n as f64) as usize % n;
                }
                adj[i * n + new_j] = 1;
            }
        }
    }

    adj
}

#[allow(clippy::needless_range_loop)]
/// Barabási-Albert scale-free network via preferential attachment.
///
/// Returns adjacency matrix as flat row-major Vec<u8> of shape [n, n].
pub fn barabasi_albert(n: usize, seed: u64) -> Vec<u8> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut adj = vec![0u8; n * n];
    let mut degree = vec![0u64; n];

    if n < 2 {
        return adj;
    }

    // Seed: node 0 ↔ node 1
    adj[1] = 1;
    adj[n] = 1;
    degree[0] = 1;
    degree[1] = 1;

    for i in 2..n {
        let total_degree: u64 = degree[..i].iter().sum();
        if total_degree == 0 {
            adj[i * n] = 1;
            degree[i] += 1;
            degree[0] += 1;
            continue;
        }

        let r = rng.random::<f64>() * total_degree as f64;
        let mut cumulative = 0.0;
        let mut target = 0;
        for j in 0..i {
            cumulative += degree[j] as f64;
            if cumulative > r {
                target = j;
                break;
            }
        }

        adj[i * n + target] = 1;
        degree[i] += 1;
        degree[target] += 1;
    }

    adj
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ws_no_self_loops() {
        let adj = watts_strogatz(20, 4, 0.3, 42);
        for i in 0..20 {
            assert_eq!(adj[i * 20 + i], 0);
        }
    }

    #[test]
    fn ws_connected() {
        let adj = watts_strogatz(10, 4, 0.0, 0);
        for i in 0..10 {
            let row_sum: u8 = (0..10).map(|j| adj[i * 10 + j]).sum();
            assert!(row_sum > 0, "node {i} has no connections");
        }
    }

    #[test]
    fn barabasi_albert_grows_correctly() {
        let adj = barabasi_albert(10, 99);
        for i in 0..10 {
            assert_eq!(adj[i * 10 + i], 0);
        }
        // Each node (after 0,1) adds at least one edge
        for i in 2..10 {
            let row_sum: u8 = (0..10).map(|j| adj[i * 10 + j]).sum();
            assert!(row_sum > 0, "node {i} has no outgoing edges");
        }
    }

    #[test]
    fn deterministic() {
        let a = watts_strogatz(10, 4, 0.3, 123);
        let b = watts_strogatz(10, 4, 0.3, 123);
        assert_eq!(a, b);
    }
}
