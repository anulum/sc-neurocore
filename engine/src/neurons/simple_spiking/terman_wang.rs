// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Terman-Wang Oscillator

//! Terman-Wang oscillatory segmentation dynamics.

/// Terman-Wang oscillator — segmentation by oscillatory correlation.
#[derive(Clone, Debug)]
pub struct TermanWangOscillator {
    pub v: f64,
    pub w: f64,
    pub alpha: f64,
    pub beta: f64,
    pub epsilon: f64,
    pub rho: f64,
    pub dt: f64,
    pub v_peak: f64,
}

impl TermanWangOscillator {
    pub fn new() -> Self {
        Self {
            v: -1.5,
            w: -0.5,
            alpha: 3.0,
            beta: 0.2,
            epsilon: 0.02,
            rho: 0.0,
            dt: 0.05,
            v_peak: 1.5,
        }
    }
    fn valid_numeric_contract(&self) -> bool {
        self.v.is_finite()
            && self.w.is_finite()
            && self.alpha.is_finite()
            && self.beta.is_finite()
            && self.epsilon.is_finite()
            && self.rho.is_finite()
            && self.dt.is_finite()
            && self.v_peak.is_finite()
            && self.beta > 0.0
            && self.epsilon > 0.0
            && self.dt > 0.0
    }
    fn derivatives(&self, v: f64, w: f64, current: f64) -> Option<(f64, f64)> {
        if !(v.is_finite() && w.is_finite() && current.is_finite()) {
            return None;
        }
        let f = 3.0 * v - v.powi(3) + 2.0;
        let g = self.alpha * (1.0 + (v / self.beta).tanh());
        let dv = f - w + current + self.rho;
        let dw = self.epsilon * (g - w);
        if dv.is_finite() && dw.is_finite() {
            Some((dv, dw))
        } else {
            None
        }
    }
    fn rk4_candidate(&self, current: f64) -> Option<(f64, f64)> {
        let dt = self.dt;
        let (k1v, k1w) = self.derivatives(self.v, self.w, current)?;
        let (k2v, k2w) =
            self.derivatives(self.v + 0.5 * dt * k1v, self.w + 0.5 * dt * k1w, current)?;
        let (k3v, k3w) =
            self.derivatives(self.v + 0.5 * dt * k2v, self.w + 0.5 * dt * k2w, current)?;
        let (k4v, k4w) = self.derivatives(self.v + dt * k3v, self.w + dt * k3w, current)?;
        let v = self.v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0;
        let w = self.w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0;
        if v.is_finite() && w.is_finite() {
            Some((v, w))
        } else {
            None
        }
    }

    fn try_step(&mut self, current: f64) -> Option<i32> {
        if !self.valid_numeric_contract() || !current.is_finite() {
            return None;
        }
        let v_prev = self.v;
        let (next_v, next_w) = self.rk4_candidate(current)?;
        self.v = next_v;
        self.w = next_w;
        Some(if self.v >= self.v_peak && v_prev < self.v_peak {
            1
        } else {
            0
        })
    }
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }
    /// Run `n_steps` RK4 updates under a constant input, returning the `v` trace
    /// and the upward-`v_peak`-crossing spike count. Reuses `step`; the cubic uses
    /// `v.powi(3)` = `v*v*v` (matching the Python `v*v*v`). The hyperbolic-tangent
    /// gating resolves to the same glibc `tanh` as the Python reference on Linux,
    /// so this backend is bit-identical there. The final state is left in
    /// `self.v` / `self.w`. Invalid input returns an empty trace and preserves
    /// the complete pre-batch state.
    pub fn simulate(&mut self, n_steps: usize, current: f64) -> (Vec<f64>, i64) {
        self.try_simulate(n_steps, current).unwrap_or_default()
    }
    /// Run one failure-atomic batch, returning `None` on any invalid stage.
    pub fn try_simulate(&mut self, n_steps: usize, current: f64) -> Option<(Vec<f64>, i64)> {
        let mut candidate = self.clone();
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes: i64 = 0;
        for _ in 0..n_steps {
            let spiked = candidate.try_step(current)?;
            trace.push(candidate.v);
            spikes += spiked as i64;
        }
        *self = candidate;
        Some((trace, spikes))
    }
    pub fn reset(&mut self) {
        self.v = -1.5;
        self.w = -0.5;
    }
}
impl Default for TermanWangOscillator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = TermanWangOscillator::default();
        let constructed = TermanWangOscillator::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn simulate_matches_repeated_step() {
        let mut simulated = TermanWangOscillator::new();
        let mut repeated = TermanWangOscillator::new();
        let (trace, spikes) = simulated.simulate(2_000, 0.5);
        let mut expected_trace = Vec::with_capacity(2_000);
        let mut expected_spikes = 0_i64;
        for _ in 0..2_000 {
            if repeated.step(0.5) == 1 {
                expected_spikes += 1;
            }
            expected_trace.push(repeated.v);
        }
        assert_eq!(trace, expected_trace);
        assert_eq!(spikes, expected_spikes);
    }

    #[test]
    fn tw_fires() {
        let mut n = TermanWangOscillator::new();
        let t: i32 = (0..2000).map(|_| n.step(0.5)).sum();
        assert!(t > 0);
    }

    #[test]
    fn tw_reset_clears_state() {
        let mut n = TermanWangOscillator::new();
        for _ in 0..500 {
            n.step(0.5);
        }
        n.reset();
        assert!((n.v - (-1.5)).abs() < 1e-10);
    }

    #[test]
    fn tw_bounded() {
        let mut n = TermanWangOscillator::new();
        for _ in 0..2000 {
            n.step(50.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn tw_matches_rk4_candidate() {
        let mut n = TermanWangOscillator::new();
        n.v = -1.2;
        n.w = -0.25;
        let current = 1.0;
        let dt = n.dt;
        let rhs = |v: f64, w: f64| {
            let f = 3.0 * v - v.powi(3) + 2.0;
            let g = n.alpha * (1.0 + (v / n.beta).tanh());
            (f - w + current + n.rho, n.epsilon * (g - w))
        };
        let (k1v, k1w) = rhs(n.v, n.w);
        let (k2v, k2w) = rhs(n.v + 0.5 * dt * k1v, n.w + 0.5 * dt * k1w);
        let (k3v, k3w) = rhs(n.v + 0.5 * dt * k2v, n.w + 0.5 * dt * k2w);
        let (k4v, k4w) = rhs(n.v + dt * k3v, n.w + dt * k3w);
        let expected = (
            n.v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0,
            n.w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0,
        );

        assert_eq!(n.step(current), 0);
        assert!((n.v - expected.0).abs() < 1e-14);
        assert!((n.w - expected.1).abs() < 1e-14);
    }

    #[test]
    fn tw_nan_no_panic() {
        let mut n = TermanWangOscillator::new();
        let before = (n.v, n.w);
        assert_eq!(n.step(f64::NAN), 0);
        assert_eq!((n.v, n.w), before);
    }

    #[test]
    fn tw_overflow_candidate_preserves_state() {
        let mut n = TermanWangOscillator {
            v: 1.0e308,
            ..Default::default()
        };
        let before = (n.v, n.w);
        assert_eq!(n.step(1.0), 0);
        assert_eq!((n.v, n.w), before);
    }

    #[test]
    fn try_simulate_rejects_overflow_without_mutation() {
        let mut neuron = TermanWangOscillator {
            v: 1.0e103,
            ..Default::default()
        };
        let before = (neuron.v, neuron.w);
        assert!(neuron.try_simulate(2, 0.5).is_none());
        assert_eq!((neuron.v, neuron.w), before);
    }

    #[test]
    fn tw_negative_no_crash() {
        let mut n = TermanWangOscillator::new();
        for _ in 0..500 {
            n.step(-5.0);
        }
        assert!(n.v.is_finite());
    }
}
