// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fractional LIF neuron model

/// Fractional-order LIF — Grünwald-Letnikov approximation. Teka et al. 2014.
#[derive(Clone, Debug)]
pub struct FractionalLIFNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub alpha: f64,
    pub resistance: f64,
    pub dt: f64,
    history: Vec<f64>,
    gl_coeffs: Vec<f64>,
    _max_hist: usize,
}

impl FractionalLIFNeuron {
    pub fn new(alpha: f64, max_hist: usize) -> Self {
        let mut coeffs = vec![0.0; max_hist + 1];
        coeffs[0] = 1.0;
        for j in 1..=max_hist {
            coeffs[j] = coeffs[j - 1] * (1.0 - (alpha + 1.0) / j as f64);
        }
        Self {
            v: 0.0,
            v_rest: 0.0,
            v_reset: 0.0,
            v_threshold: 1.0,
            alpha,
            resistance: 1.0,
            dt: 1.0,
            history: vec![0.0; max_hist],
            gl_coeffs: coeffs,
            _max_hist: max_hist,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        // Grünwald-Letnikov: D^α v ≈ (1/dt^α) Σ_j c_j v(t-j·dt)
        let mut gl_sum = 0.0;
        let n = self.history.len().min(self.gl_coeffs.len() - 1);
        for j in 0..n {
            gl_sum += self.gl_coeffs[j + 1] * self.history[n - 1 - j];
        }
        let rhs = -(self.v - self.v_rest) + self.resistance * current;
        self.v = rhs * self.dt.powf(self.alpha) - gl_sum;
        // Shift history
        let len = self.history.len();
        if len > 0 {
            for i in 0..len - 1 {
                self.history[i] = self.history[i + 1];
            }
            self.history[len - 1] = self.v;
        }
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.history.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frac_lif_fires() {
        let mut n = FractionalLIFNeuron::new(0.8, 50);
        let t: i32 = (0..200).map(|_| n.step(2.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn frac_lif_reset() {
        let mut n = FractionalLIFNeuron::new(0.8, 50);
        for _ in 0..100 {
            n.step(2.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
    }

    #[test]
    fn frac_lif_nan_no_panic() {
        FractionalLIFNeuron::new(0.8, 50).step(f64::NAN);
    }
}
