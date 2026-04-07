// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Floating-point E-I balanced network for Studio

//! Fused E-I balanced LIF network simulation in f64 arithmetic.
//!
//! Runs a complete excitatory-inhibitory network with CSR weight matrix,
//! Poisson external drive, and spike-scatter coupling in a single Rust
//! call — no per-step Python overhead.

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Result of an E-I network simulation.
pub struct EIResult {
    pub spike_times: Vec<f64>,
    pub spike_neurons: Vec<u32>,
    pub n_exc: u32,
    pub n_inh: u32,
    pub rate_time: Vec<f64>,
    pub exc_rates: Vec<f64>,
    pub inh_rates: Vec<f64>,
    pub mean_exc_rate: f64,
    pub mean_inh_rate: f64,
}

/// Run a complete E-I balanced LIF network simulation.
///
/// All computation happens in Rust — connectivity build, Poisson input,
/// Euler integration, spike detection, rate binning.
#[allow(clippy::too_many_arguments)]
pub fn simulate_ei(
    n_exc: usize,
    n_inh: usize,
    w_ee: f64,
    w_ei: f64,
    w_ie: f64,
    w_ii: f64,
    p_conn: f64,
    ext_rate: f64,
    duration: f64,
    dt: f64,
    seed: u64,
) -> EIResult {
    let n = n_exc + n_inh;
    let n_steps = ((duration / dt) as usize).min(50_000);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    // LIF params
    let tau_m = 20.0_f64;
    let v_rest = -65.0_f64;
    let v_threshold = -50.0_f64;
    let v_reset = -65.0_f64;
    let tau_ref = 2.0_f64;

    // Build CSR weight matrix
    let mut row_offsets = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_offsets.push(0usize);

    for i in 0..n {
        let i_exc = i < n_exc;
        for j in 0..n {
            if i == j {
                continue;
            }
            if rng.random::<f64>() >= p_conn {
                continue;
            }
            let j_exc = j < n_exc;
            let w = match (i_exc, j_exc) {
                (true, true) => w_ee,
                (true, false) => w_ie,
                (false, true) => -w_ei,
                (false, false) => -w_ii,
            };
            col_indices.push(j);
            values.push(w);
        }
        row_offsets.push(col_indices.len());
    }

    // State arrays
    let mut v = vec![v_rest; n];
    let mut refractory = vec![0.0_f64; n];
    let mut prev_spiked = vec![false; n];

    // Recording
    let mut spike_times: Vec<f64> = Vec::new();
    let mut spike_neurons: Vec<u32> = Vec::new();
    let bin_size = (n_steps / 100).max(1);
    let n_bins = n_steps / bin_size;
    let mut exc_rates = vec![0.0_f64; n_bins];
    let mut inh_rates = vec![0.0_f64; n_bins];
    let mut exc_bin = 0u32;
    let mut inh_bin = 0u32;

    let ext_lambda = (ext_rate * dt / 1000.0).max(0.0);
    let exp_neg_lambda = (-ext_lambda).exp();

    for t in 0..n_steps {
        // Decay refractory
        for r in refractory.iter_mut() {
            *r = (*r - dt).max(0.0);
        }

        // Synaptic input from previous spikes (CSR scatter)
        let mut syn = vec![0.0_f64; n];
        for i in 0..n {
            if !prev_spiked[i] {
                continue;
            }
            let start = row_offsets[i];
            let end = row_offsets[i + 1];
            for k in start..end {
                syn[col_indices[k]] += values[k];
            }
        }

        // External Poisson + Euler step
        for i in 0..n {
            if refractory[i] > 0.0 {
                continue;
            }
            // Knuth Poisson sampling (fast for small lambda)
            let mut k = 0u32;
            let mut p = 1.0_f64;
            loop {
                p *= rng.random::<f64>();
                if p <= exp_neg_lambda {
                    break;
                }
                k += 1;
            }
            let ext = k as f64 * 5.0;
            let dv = (-(v[i] - v_rest) / tau_m + ext + syn[i]) * dt;
            v[i] += dv;
        }

        // Spike detection
        prev_spiked.fill(false);
        for i in 0..n {
            if refractory[i] > 0.0 {
                continue;
            }
            if v[i] >= v_threshold {
                v[i] = v_reset;
                refractory[i] = tau_ref;
                prev_spiked[i] = true;
                spike_times.push(t as f64 * dt);
                spike_neurons.push(i as u32);
                if i < n_exc {
                    exc_bin += 1;
                } else {
                    inh_bin += 1;
                }
            }
        }

        // Rate binning
        if (t + 1) % bin_size == 0 {
            let bi = t / bin_size;
            if bi < n_bins {
                let bin_t = bin_size as f64 * dt / 1000.0;
                exc_rates[bi] = exc_bin as f64 / n_exc.max(1) as f64 / bin_t.max(0.001);
                inh_rates[bi] = inh_bin as f64 / n_inh.max(1) as f64 / bin_t.max(0.001);
            }
            exc_bin = 0;
            inh_bin = 0;
        }
    }

    let rate_time: Vec<f64> = (0..n_bins)
        .map(|i| i as f64 * bin_size as f64 * dt)
        .collect();

    let mean_exc = if exc_rates.iter().any(|&r| r > 0.0) {
        let pos: Vec<f64> = exc_rates.iter().copied().filter(|&r| r > 0.0).collect();
        pos.iter().sum::<f64>() / pos.len() as f64
    } else {
        0.0
    };
    let mean_inh = if inh_rates.iter().any(|&r| r > 0.0) {
        let pos: Vec<f64> = inh_rates.iter().copied().filter(|&r| r > 0.0).collect();
        pos.iter().sum::<f64>() / pos.len() as f64
    } else {
        0.0
    };

    EIResult {
        spike_times,
        spike_neurons,
        n_exc: n_exc as u32,
        n_inh: n_inh as u32,
        rate_time,
        exc_rates,
        inh_rates,
        mean_exc_rate: (mean_exc * 10.0).round() / 10.0,
        mean_inh_rate: (mean_inh * 10.0).round() / 10.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ei_network_runs_without_panic() {
        let r = simulate_ei(20, 5, 0.1, 0.4, 0.1, 0.4, 0.2, 10.0, 50.0, 0.1, 42);
        assert_eq!(r.n_exc, 20);
        assert_eq!(r.n_inh, 5);
        assert!(!r.rate_time.is_empty());
        assert_eq!(r.exc_rates.len(), r.rate_time.len());
    }

    #[test]
    fn ei_network_with_high_drive_produces_spikes() {
        // ext_rate=5000 Hz → lambda=0.5 per step → frequent Poisson events
        let r = simulate_ei(40, 10, 0.1, 0.4, 0.1, 0.4, 0.2, 5000.0, 100.0, 0.1, 42);
        assert!(
            !r.spike_times.is_empty(),
            "high drive should produce spikes"
        );
    }
}
