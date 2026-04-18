// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Canonical cortical microcircuit in Rust

//! 5-population cortical column (Douglas & Martin 2004).
//! L4 (thalamic) → L2/3 exc ↔ L2/3 inh → L5 (output) → L6 (feedback → L4).

/// Cortical column with dense weight matrices and LIF-like dynamics.
pub struct CorticalColumnRust {
    pub n: usize,
    decay: f64,
    threshold: f64,
    dt_over_tau: f64,
    // Membrane voltages
    pub v_l4: Vec<f64>,
    pub v_l23e: Vec<f64>,
    pub v_l23i: Vec<f64>,
    pub v_l5: Vec<f64>,
    pub v_l6: Vec<f64>,
    // Weight matrices (row-major, n×n each)
    w_thal_l4: Vec<f64>,
    w_l4_l23e: Vec<f64>,
    w_l23e_l23i: Vec<f64>,
    w_l23i_l23e: Vec<f64>,
    w_l23e_l5: Vec<f64>,
    w_l5_l6: Vec<f64>,
    w_l6_l4: Vec<f64>,
}

impl CorticalColumnRust {
    pub fn new(
        n: usize,
        tau: f64,
        dt: f64,
        threshold: f64,
        w_exc: f64,
        w_inh: f64,
        seed: u64,
    ) -> Self {
        let decay = (-dt / tau).exp();
        let dt_over_tau = dt / tau;
        let mut rng = SimpleRng::new(seed);

        let make_weights = |rng: &mut SimpleRng, strength: f64, prob: f64| -> Vec<f64> {
            let mut w = vec![0.0f64; n * n];
            for i in 0..n * n {
                if rng.next_f64() < prob {
                    w[i] = rng.next_f64() * strength.abs();
                    if strength < 0.0 {
                        w[i] = -w[i];
                    }
                }
            }
            w
        };

        Self {
            n,
            decay,
            threshold,
            dt_over_tau,
            v_l4: vec![0.0; n],
            v_l23e: vec![0.0; n],
            v_l23i: vec![0.0; n],
            v_l5: vec![0.0; n],
            v_l6: vec![0.0; n],
            w_thal_l4: make_weights(&mut rng, w_exc, 0.5),
            w_l4_l23e: make_weights(&mut rng, w_exc, 0.4),
            w_l23e_l23i: make_weights(&mut rng, w_exc, 0.3),
            w_l23i_l23e: make_weights(&mut rng, w_inh, 0.3),
            w_l23e_l5: make_weights(&mut rng, w_exc, 0.3),
            w_l5_l6: make_weights(&mut rng, w_exc, 0.3),
            w_l6_l4: make_weights(&mut rng, w_exc * 0.5, 0.2),
        }
    }

    /// Run one timestep. Returns spike vectors for all 5 populations.
    pub fn step(&mut self, thalamic_input: &[f64]) -> [Vec<f64>; 5] {
        let n = self.n;

        // Helper: matrix-vector multiply (row-major n×n × n → n)
        let matvec = |w: &[f64], x: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0; n];
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += w[i * n + j] * x[j];
                }
                out[i] = sum;
            }
            out
        };

        let thresh_vec = |v: &[f64]| -> Vec<f64> {
            v.iter()
                .map(|&vi| if vi > self.threshold { 1.0 } else { 0.0 })
                .collect()
        };

        // L6 feedback spikes (from previous step)
        let l6_spk = thresh_vec(&self.v_l6);

        // L4: thalamic + L6 feedback
        let i_l4_thal = matvec(&self.w_thal_l4, thalamic_input);
        let i_l4_fb = matvec(&self.w_l6_l4, &l6_spk);
        for i in 0..n {
            self.v_l4[i] =
                self.decay * self.v_l4[i] + (i_l4_thal[i] + i_l4_fb[i]) * self.dt_over_tau;
        }
        let spk_l4 = thresh_vec(&self.v_l4);
        for i in 0..n {
            self.v_l4[i] -= spk_l4[i] * self.threshold;
        }

        // L2/3 exc
        let i_l23e_ff = matvec(&self.w_l4_l23e, &spk_l4);
        let l23i_spk = thresh_vec(&self.v_l23i);
        let i_l23e_inh = matvec(&self.w_l23i_l23e, &l23i_spk);
        for i in 0..n {
            self.v_l23e[i] =
                self.decay * self.v_l23e[i] + (i_l23e_ff[i] + i_l23e_inh[i]) * self.dt_over_tau;
        }
        let spk_l23e = thresh_vec(&self.v_l23e);
        for i in 0..n {
            self.v_l23e[i] -= spk_l23e[i] * self.threshold;
        }

        // L2/3 inh
        let i_l23i = matvec(&self.w_l23e_l23i, &spk_l23e);
        for i in 0..n {
            self.v_l23i[i] = self.decay * self.v_l23i[i] + i_l23i[i] * self.dt_over_tau;
        }
        let spk_l23i = thresh_vec(&self.v_l23i);
        for i in 0..n {
            self.v_l23i[i] -= spk_l23i[i] * self.threshold;
        }

        // L5
        let i_l5 = matvec(&self.w_l23e_l5, &spk_l23e);
        for i in 0..n {
            self.v_l5[i] = self.decay * self.v_l5[i] + i_l5[i] * self.dt_over_tau;
        }
        let spk_l5 = thresh_vec(&self.v_l5);
        for i in 0..n {
            self.v_l5[i] -= spk_l5[i] * self.threshold;
        }

        // L6
        let i_l6 = matvec(&self.w_l5_l6, &spk_l5);
        for i in 0..n {
            self.v_l6[i] = self.decay * self.v_l6[i] + i_l6[i] * self.dt_over_tau;
        }
        let spk_l6_new = thresh_vec(&self.v_l6);
        for i in 0..n {
            self.v_l6[i] -= spk_l6_new[i] * self.threshold;
        }

        [spk_l4, spk_l23e, spk_l23i, spk_l5, spk_l6_new]
    }

    pub fn reset(&mut self) {
        self.v_l4.fill(0.0);
        self.v_l23e.fill(0.0);
        self.v_l23i.fill(0.0);
        self.v_l5.fill(0.0);
        self.v_l6.fill(0.0);
    }
}

/// Minimal deterministic PRNG for weight initialization.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_step_output_lengths() {
        let mut col = CorticalColumnRust::new(10, 10.0, 1.0, 1.0, 0.5, -0.3, 42);
        let input = vec![5.0; 10];
        let spikes = col.step(&input);
        assert_eq!(spikes.len(), 5);
        for pop in &spikes {
            assert_eq!(pop.len(), 10);
        }
    }

    #[test]
    fn test_column_produces_spikes() {
        let mut col = CorticalColumnRust::new(20, 10.0, 1.0, 0.5, 1.0, -0.3, 42);
        let input = vec![10.0; 20];
        let mut total = 0.0;
        for _ in 0..50 {
            let spikes = col.step(&input);
            total += spikes[0].iter().sum::<f64>(); // L4 spikes
        }
        assert!(total > 0.0, "Expected L4 spikes");
    }

    #[test]
    fn test_column_reset() {
        let mut col = CorticalColumnRust::new(5, 10.0, 1.0, 1.0, 0.5, -0.3, 42);
        let input = vec![10.0; 5];
        col.step(&input);
        col.reset();
        assert!(col.v_l4.iter().all(|&v| v == 0.0));
        assert!(col.v_l5.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_column_deterministic() {
        let mut col_a = CorticalColumnRust::new(5, 10.0, 1.0, 1.0, 0.5, -0.3, 99);
        let mut col_b = CorticalColumnRust::new(5, 10.0, 1.0, 1.0, 0.5, -0.3, 99);
        let input = vec![3.0; 5];
        let sa = col_a.step(&input);
        let sb = col_b.step(&input);
        assert_eq!(sa, sb);
    }
}
