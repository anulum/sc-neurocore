// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
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
//!
//! Synapses are delta synapses in the sense of Brunel (2000, J. Comput.
//! Neurosci. 8:183): a presynaptic spike moves the postsynaptic membrane by
//! its weight in millivolts, one step later. Every neuron also receives
//! [`EXTERNAL_SYNAPSES`] independent excitatory Poisson inputs, each firing at
//! `ext_rate` Hz and moving the membrane by [`EXTERNAL_WEIGHT_MV`]. The
//! external drive alone reaches threshold at about 9.4 Hz per input
//! (15 mV / (0.1 mV x 800 x 20 ms)).
//!
//! An earlier version multiplied each jump by the step size, so a spike moved
//! the membrane by `weight x dt` (0.01 mV for the default E-to-E weight at
//! dt = 0.1 ms), and the external drive was one input per neuron: no setting
//! of the Studio's controls produced a single spike.

use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Poisson};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Independent external excitatory Poisson inputs per neuron.
pub const EXTERNAL_SYNAPSES: f64 = 800.0;

/// Membrane jump caused by one external input spike, in millivolts.
pub const EXTERNAL_WEIGHT_MV: f64 = 0.1;

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

    // Expected external input spikes per neuron per step. `Poisson` samples
    // exactly for any mean; the multiply-uniforms loop it replaces never
    // terminated once exp(-mean) underflowed to zero.
    let ext_lambda = EXTERNAL_SYNAPSES * ext_rate.max(0.0) * dt / 1000.0;
    let ext_inputs = if ext_lambda > 0.0 {
        Poisson::new(ext_lambda).ok()
    } else {
        None
    };
    let mut exc_spikes = 0usize;
    let mut inh_spikes = 0usize;

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
            let external = match &ext_inputs {
                Some(poisson) => poisson.sample(&mut rng) * EXTERNAL_WEIGHT_MV,
                None => 0.0,
            };
            v[i] += -(v[i] - v_rest) / tau_m * dt + external + syn[i];
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
                    exc_spikes += 1;
                } else {
                    inh_bin += 1;
                    inh_spikes += 1;
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

    // Mean rate per neuron over the simulated time. It used to average only
    // the rate bins that held a spike, which overstated sparse activity.
    let simulated_s = (n_steps as f64 * dt / 1000.0).max(f64::MIN_POSITIVE);
    let mean_exc = exc_spikes as f64 / n_exc.max(1) as f64 / simulated_s;
    let mean_inh = inh_spikes as f64 / n_inh.max(1) as f64 / simulated_s;

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

    const DEFAULT: (f64, f64, f64, f64, f64) = (0.1, 0.4, 0.1, 0.4, 0.2);

    fn run(ext_rate: f64, duration: f64, dt: f64) -> EIResult {
        let (w_ee, w_ei, w_ie, w_ii, p_conn) = DEFAULT;
        simulate_ei(
            80, 20, w_ee, w_ei, w_ie, w_ii, p_conn, ext_rate, duration, dt, 42,
        )
    }

    #[test]
    fn ei_network_runs_without_panic() {
        let r = simulate_ei(20, 5, 0.1, 0.4, 0.1, 0.4, 0.2, 10.0, 50.0, 0.1, 42);
        assert_eq!(r.n_exc, 20);
        assert_eq!(r.n_inh, 5);
        assert!(!r.rate_time.is_empty());
        assert_eq!(r.exc_rates.len(), r.rate_time.len());
    }

    #[test]
    fn default_drive_above_threshold_fires() {
        // 12 Hz per input is 1.3x the 9.4 Hz threshold rate of the drive.
        let r = run(12.0, 500.0, 0.1);
        assert!(
            r.mean_exc_rate > 5.0 && r.mean_exc_rate < 100.0,
            "{}",
            r.mean_exc_rate
        );
        assert!(
            r.mean_inh_rate > 5.0 && r.mean_inh_rate < 100.0,
            "{}",
            r.mean_inh_rate
        );
    }

    #[test]
    fn drive_well_below_threshold_stays_silent() {
        // Mean drive 3.2 mV against a 15 mV gap, fluctuations of about 0.3 mV.
        assert!(run(2.0, 500.0, 0.1).spike_times.is_empty());
    }

    #[test]
    fn rate_does_not_depend_on_the_step_size() {
        // A spike moves the membrane by its weight, not by weight x dt.
        let coarse = run(12.0, 1000.0, 0.1).mean_exc_rate;
        let fine = run(12.0, 1000.0, 0.05).mean_exc_rate;
        assert!(
            (coarse - fine).abs() < 0.2 * coarse.max(fine),
            "{coarse} vs {fine}"
        );
    }

    #[test]
    fn mean_rate_counts_silent_time() {
        let r = run(12.0, 500.0, 0.1);
        let spikes = r.spike_neurons.iter().filter(|&&i| i < 80).count() as f64;
        assert!((r.mean_exc_rate - spikes / 80.0 / 0.5).abs() < 0.051);
    }

    #[test]
    fn very_large_external_mean_terminates() {
        // 800 inputs x 100 Hz x 5 ms = 400 expected events per step.
        let r = run(100.0, 50.0, 5.0);
        assert!(!r.spike_times.is_empty());
    }
}
