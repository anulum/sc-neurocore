// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fused Brunel balanced network simulation in fixed-point

//! Fused Brunel balanced network simulation in fixed-point Q8.8 arithmetic.
//!
//! Runs a complete Brunel (2000) network with CSR weight matrix, Poisson
//! external drive, and recurrent spike-scatter coupling in a single Rust
//! call — no per-step Python overhead.

use crate::neuron::{mask, FixedPointLif};
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Brunel balanced network with CSR connectivity and Poisson drive.
pub struct BrunelNetwork {
    neurons: Vec<FixedPointLif>,
    prev_spikes: Vec<bool>,
    // CSR weight matrix (Q8.8 fixed-point values)
    w_row_offsets: Vec<usize>,
    w_col_indices: Vec<usize>,
    w_values: Vec<i16>,
    n_neurons: usize,
    leak_k: i16,
    gain_k: i16,
    ext_lambda: f64,
    ext_weight_fp: i16,
    poisson: Option<Poisson<f64>>,
    rng: Xoshiro256PlusPlus,
}

impl BrunelNetwork {
    /// Build from CSR arrays and neuron parameters.
    ///
    /// `w_row_offsets`: length n_neurons+1
    /// `w_col_indices`, `w_values`: length nnz, values in Q8.8
    ///
    /// # Errors
    ///
    /// Rejects invalid CSR bounds/order, a neuron count exceeding the `u32`
    /// spike-count range, widths outside `1..=16`, fractions outside the width,
    /// negative refractory periods, and invalid Poisson means. Zero drive is
    /// accepted; positive means must fit `Poisson::<f64>::MAX_LAMBDA`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_neurons: usize,
        w_row_offsets: Vec<usize>,
        w_col_indices: Vec<usize>,
        w_values: Vec<i16>,
        data_width: u32,
        fraction: u32,
        v_rest: i16,
        v_reset: i16,
        v_threshold: i16,
        refractory_period: i32,
        leak_k: i16,
        gain_k: i16,
        ext_lambda: f64,
        ext_weight_fp: i16,
        seed: u64,
    ) -> Result<Self, String> {
        if n_neurons > u32::MAX as usize {
            return Err("n_neurons must not exceed u32::MAX".to_string());
        }
        let row_count = n_neurons
            .checked_add(1)
            .ok_or_else(|| "n_neurons+1 exceeds usize range".to_string())?;
        if w_row_offsets.len() != row_count {
            return Err(format!(
                "w_row_offsets length {} != n_neurons+1={}",
                w_row_offsets.len(),
                row_count
            ));
        }
        if w_col_indices.len() != w_values.len() {
            return Err(format!(
                "w_col_indices len {} != w_values len {}",
                w_col_indices.len(),
                w_values.len()
            ));
        }
        if w_row_offsets[0] != 0 {
            return Err("w_row_offsets must start at 0".to_string());
        }
        if w_row_offsets[n_neurons] != w_values.len() {
            return Err("w_row_offsets must end at the number of weights".to_string());
        }
        if w_row_offsets.windows(2).any(|row| row[0] > row[1]) {
            return Err("w_row_offsets must be nondecreasing".to_string());
        }
        if w_col_indices.iter().any(|&column| column >= n_neurons) {
            return Err("w_col_indices must be less than n_neurons".to_string());
        }
        if !(1..=16).contains(&data_width) {
            return Err("data_width must be in [1, 16]".to_string());
        }
        if fraction >= data_width {
            return Err("fraction must be less than data_width".to_string());
        }
        if refractory_period < 0 {
            return Err("refractory_period must be nonnegative".to_string());
        }
        if !ext_lambda.is_finite() || ext_lambda < 0.0 {
            return Err("ext_lambda must be finite and nonnegative".to_string());
        }
        let poisson = if ext_lambda >= 30.0 {
            Some(
                Poisson::<f64>::new(ext_lambda)
                    .map_err(|error| format!("Invalid ext_lambda: {error}"))?,
            )
        } else {
            None
        };
        let neurons: Vec<FixedPointLif> = (0..n_neurons)
            .map(|_| {
                FixedPointLif::new(
                    data_width,
                    fraction,
                    v_rest,
                    v_reset,
                    v_threshold,
                    refractory_period,
                )
            })
            .collect();

        Ok(Self {
            neurons,
            prev_spikes: vec![false; n_neurons],
            w_row_offsets,
            w_col_indices,
            w_values,
            n_neurons,
            leak_k,
            gain_k,
            ext_lambda,
            ext_weight_fp,
            poisson,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        })
    }

    /// Run the simulation and return per-step `u32` spike counts.
    ///
    /// Signed currents wrap before width masking. Means below 30 retain the
    /// seeded Knuth stream; larger means use `rand_distr` rejection sampling.
    pub fn run(&mut self, n_steps: usize) -> Vec<u32> {
        let n = self.n_neurons;
        let mut i_syn = vec![0i32; n];
        let mut counts = Vec::with_capacity(n_steps);

        for _ in 0..n_steps {
            // 1. CSR spike-scatter: for each fired pre-synaptic neuron,
            //    scatter its outgoing weights to post-synaptic current buffer.
            i_syn.iter_mut().for_each(|x| *x = 0);
            for pre in 0..n {
                if !self.prev_spikes[pre] {
                    continue;
                }
                let start = self.w_row_offsets[pre];
                let end = self.w_row_offsets[pre + 1];
                for idx in start..end {
                    let post = self.w_col_indices[idx];
                    i_syn[post] = i_syn[post].wrapping_add(self.w_values[idx] as i32);
                }
            }

            // 2. Poisson external input + LIF step per neuron
            let mut step_spikes = 0u32;
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let ext_count = match &self.poisson {
                    Some(distribution) => distribution.sample(&mut self.rng) as u64,
                    None => poisson_sample(&mut self.rng, self.ext_lambda) as u64,
                };
                let ext_current = (ext_count as i32).wrapping_mul(self.ext_weight_fp as i32);
                let total_current = i_syn[i].wrapping_add(ext_current);
                let dw = self.neurons[i].data_width;
                let i_t = mask(total_current, dw);

                let (spike, _) = self.neurons[i].step(self.leak_k, self.gain_k, i_t, 0);
                self.prev_spikes[i] = spike > 0;
                if spike > 0 {
                    step_spikes += 1;
                }
            }
            counts.push(step_spikes);
        }
        counts
    }

    pub fn total_spikes(&self, counts: &[u32]) -> u64 {
        counts.iter().map(|&c| c as u64).sum()
    }
}

/// Knuth's algorithm for Poisson sampling (λ < 30).
fn poisson_sample(rng: &mut Xoshiro256PlusPlus, lambda: f64) -> u32 {
    if lambda <= 0.0 {
        return 0;
    }
    use rand::RngExt;
    let l = (-lambda).exp();
    let mut k = 0u32;
    let mut p = 1.0f64;
    loop {
        k += 1;
        p *= rng.random::<f64>();
        if p <= l {
            return k - 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_small_network() -> BrunelNetwork {
        // 4 neurons, all-to-all excitatory (weight=26 in Q8.8 ≈ 0.1)
        let n = 4;
        let mut row_offsets = vec![0usize; n + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    col_indices.push(j);
                    values.push(26i16); // ~0.1 in Q8.8
                }
            }
            row_offsets[i + 1] = col_indices.len();
        }

        BrunelNetwork::new(
            n,
            row_offsets,
            col_indices,
            values,
            16,
            8, // data_width, fraction
            0,
            0,
            256, // v_rest, v_reset, v_threshold (1.0 in Q8.8)
            2,   // refractory_period
            1,   // leak_k
            256, // gain_k (1.0 in Q8.8)
            5.0, // ext_lambda (Poisson spikes/step)
            26,  // ext_weight_fp (~0.1 in Q8.8)
            42,
        )
        .unwrap()
    }

    #[test]
    fn brunel_produces_spikes() {
        let mut net = make_small_network();
        let counts = net.run(100);
        let total: u64 = net.total_spikes(&counts);
        assert!(total > 0, "network must produce spikes");
    }

    #[test]
    fn brunel_empty_network() {
        let mut net = BrunelNetwork::new(
            0,
            vec![0],
            vec![],
            vec![],
            16,
            8,
            0,
            0,
            256,
            2,
            1,
            256,
            0.0,
            0,
            42,
        )
        .unwrap();
        let counts = net.run(10);
        assert!(counts.iter().all(|&c| c == 0));
    }

    #[test]
    fn brunel_csr_validation() {
        let result = BrunelNetwork::new(
            4,
            vec![0, 1],
            vec![0],
            vec![10],
            16,
            8,
            0,
            0,
            256,
            2,
            1,
            256,
            0.0,
            0,
            42,
        );
        assert!(result.is_err());
    }
}
