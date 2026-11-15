// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fractional LIF neuron model

#![deny(missing_docs)]

/// Fractional-order LIF — Grünwald-Letnikov approximation. Teka et al. 2014.
#[derive(Clone, Debug)]
pub struct FractionalLIFNeuron {
    /// Membrane potential (dimensionless).
    pub v: f64,
    /// Resting / leak reversal potential (dimensionless).
    pub v_rest: f64,
    /// Membrane potential reset to after a spike (dimensionless).
    pub v_reset: f64,
    /// Voltage at which a spike is registered (dimensionless).
    pub v_threshold: f64,
    /// Fractional-derivative order (memory exponent; 1 recovers the standard LIF) (dimensionless).
    pub alpha: f64,
    /// Normalised membrane resistance (R=1 in the project's normalised-current convention) (dimensionless).
    pub resistance: f64,
    /// Step; the drive is scaled by `dt^alpha`.
    pub dt: f64,
    history: Vec<f64>,
    gl_coeffs: Vec<f64>,
    _max_hist: usize,
}

impl FractionalLIFNeuron {
    /// Create a neuron at rest with fractional order `alpha` and `max_hist` steps of memory.
    ///
    /// The Grünwald-Letnikov weights `c_j = c_(j-1) * (1 - (alpha + 1) / j)` are computed
    /// here once, so a later change of the `alpha` field does not change them.
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
    /// Advance one step with the input `current` and return 1 on a spike, else 0.
    ///
    /// `v` becomes `(-(v - v_rest) + resistance * current) * dt^alpha - sum_j c_j v_(n-j)`
    /// over the stored history, all `max_hist` values of it. Reaching `v_threshold` sets `v`
    /// to `v_reset`; the history then stores `v` after any reset, so the memory holds the
    /// reset value, as in the Python model. The input is not validated.
    pub fn step(&mut self, current: f64) -> i32 {
        // Grünwald-Letnikov: D^α v ≈ (1/dt^α) Σ_j c_j v(t-j·dt)
        let mut gl_sum = 0.0;
        let n = self.history.len().min(self.gl_coeffs.len() - 1);
        for j in 0..n {
            gl_sum += self.gl_coeffs[j + 1] * self.history[n - 1 - j];
        }
        let rhs = -(self.v - self.v_rest) + self.resistance * current;
        self.v = rhs * self.dt.powf(self.alpha) - gl_sum;
        let spiked = self.v >= self.v_threshold;
        if spiked {
            self.v = self.v_reset;
        }
        // The memory keeps the potential after the reset, as the Python model does; keeping the
        // value before it let the memory term undo every reset.
        let len = self.history.len();
        if len > 0 {
            for i in 0..len - 1 {
                self.history[i] = self.history[i + 1];
            }
            self.history[len - 1] = self.v;
        }
        i32::from(spiked)
    }
    /// Return `v` to `v_rest` and fill the history with `v_rest`; the weights and the
    /// parameters are kept.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.history.fill(self.v_rest);
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
    fn frac_lif_reset_holds_against_a_subthreshold_drive() {
        // One supra-threshold step, then a drive below threshold. A plain Euler LIF with
        // dt = tau = 1 settles at the drive, 0.5, and alpha = 1 must do the same.
        let mut n = FractionalLIFNeuron::new(1.0, 50);
        assert_eq!(n.step(1.5), 1);
        for _ in 0..6 {
            assert_eq!(n.step(0.5), 0);
            assert!((n.v - 0.5).abs() < 1e-12, "v = {}", n.v);
        }
    }

    #[test]
    fn frac_lif_memory_stores_the_reset_value() {
        let mut n = FractionalLIFNeuron::new(0.8, 50);
        assert_eq!(n.step(1.5), 1);
        assert_eq!(n.history[49], n.v_reset);
        // With the reset value in memory: 0.5, then 0.5 - 0.5 + 0.8 * 0.5 = 0.4.
        assert_eq!(n.step(0.5), 0);
        assert!((n.v - 0.5).abs() < 1e-12, "v = {}", n.v);
        assert_eq!(n.step(0.5), 0);
        assert!((n.v - 0.4).abs() < 1e-12, "v = {}", n.v);
    }

    #[test]
    fn frac_lif_reset_fills_the_memory_with_v_rest() {
        let mut n = FractionalLIFNeuron::new(0.8, 10);
        n.v_rest = -0.25;
        for _ in 0..20 {
            n.step(0.7);
        }
        n.reset();
        assert!(n.history.iter().all(|&h| h == -0.25));
        assert_eq!(n.v, -0.25);
    }

    #[test]
    fn frac_lif_nan_no_panic() {
        FractionalLIFNeuron::new(0.8, 50).step(f64::NAN);
    }
}
