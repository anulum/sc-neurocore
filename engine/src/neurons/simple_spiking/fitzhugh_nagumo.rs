// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — FitzHugh-Nagumo Neuron Model

//! FitzHugh-Nagumo qualitative spiking dynamics.

/// FitzHugh-Nagumo 1961 — 2D qualitative spike model.
#[derive(Clone, Debug)]
pub struct FitzHughNagumoNeuron {
    pub v: f64,
    pub w: f64,
    pub a: f64,
    pub b: f64,
    pub epsilon: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl FitzHughNagumoNeuron {
    pub fn new() -> Self {
        Self {
            v: -1.0,
            w: -0.5,
            a: 0.7,
            b: 0.8,
            epsilon: 0.08,
            dt: 0.1,
            v_threshold: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !(current.is_finite() && self.is_valid()) {
            return -1;
        }
        let v_prev = self.v;
        let Some((new_v, new_w)) = self.rk4_candidate(current) else {
            return -1;
        };
        if !(new_v.is_finite() && new_w.is_finite()) {
            return -1;
        }
        self.v = new_v;
        self.w = new_w;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = -1.0;
        self.w = -0.5;
    }

    /// Run `n_steps` RK4 updates under a constant input, returning the `v` trace
    /// and the upward-crossing spike count. Reuses `step` (RK4) so the trace is
    /// bit-identical to the per-step path and — because the right-hand side is
    /// exact arithmetic (`v.powi(3)` is `v*v*v`) — to the Python reference. The
    /// final state is left in `self.v` / `self.w`.
    pub fn simulate(&mut self, n_steps: usize, current: f64) -> (Vec<f64>, i64) {
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes: i64 = 0;
        for _ in 0..n_steps {
            let spiked = self.step(current);
            trace.push(self.v);
            if spiked == 1 {
                spikes += 1;
            }
        }
        (trace, spikes)
    }

    fn is_valid(&self) -> bool {
        self.v.is_finite()
            && self.w.is_finite()
            && self.a.is_finite()
            && self.b.is_finite()
            && self.epsilon.is_finite()
            && self.dt.is_finite()
            && self.v_threshold.is_finite()
            && self.b > 0.0
            && self.epsilon > 0.0
            && self.dt > 0.0
    }

    fn rhs(&self, v: f64, w: f64, current: f64) -> Option<(f64, f64)> {
        if !(v.is_finite() && w.is_finite() && current.is_finite()) {
            return None;
        }
        let dv = v - v.powi(3) / 3.0 - w + current;
        let dw = self.epsilon * (v + self.a - self.b * w);
        if dv.is_finite() && dw.is_finite() {
            Some((dv, dw))
        } else {
            None
        }
    }

    fn rk4_candidate(&self, current: f64) -> Option<(f64, f64)> {
        let (v0, w0, dt) = (self.v, self.w, self.dt);
        let (k1v, k1w) = self.rhs(v0, w0, current)?;
        let (k2v, k2w) = self.rhs(v0 + 0.5 * dt * k1v, w0 + 0.5 * dt * k1w, current)?;
        let (k3v, k3w) = self.rhs(v0 + 0.5 * dt * k2v, w0 + 0.5 * dt * k2w, current)?;
        let (k4v, k4w) = self.rhs(v0 + dt * k3v, w0 + dt * k3w, current)?;
        Some((
            v0 + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0,
            w0 + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0,
        ))
    }
}
impl Default for FitzHughNagumoNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = FitzHughNagumoNeuron::default();
        let constructed = FitzHughNagumoNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn simulate_matches_repeated_step() {
        let mut simulated = FitzHughNagumoNeuron::new();
        let mut repeated = FitzHughNagumoNeuron::new();
        let (trace, spikes) = simulated.simulate(2_000, 1.0);
        let mut expected_trace = Vec::with_capacity(2_000);
        let mut expected_spikes = 0_i64;
        for _ in 0..2_000 {
            if repeated.step(1.0) == 1 {
                expected_spikes += 1;
            }
            expected_trace.push(repeated.v);
        }
        assert_eq!(trace, expected_trace);
        assert_eq!(spikes, expected_spikes);
    }

    #[test]
    fn fhn_fires() {
        let mut n = FitzHughNagumoNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(1.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn fhn_silent_without_input() {
        let mut n = FitzHughNagumoNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }

    #[test]
    fn fhn_reset_clears_state() {
        let mut n = FitzHughNagumoNeuron::new();
        for _ in 0..500 {
            n.step(1.0);
        }
        n.reset();
        assert!((n.v - (-1.0)).abs() < 1e-10);
        assert!((n.w - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn fhn_moderate_input_stable() {
        let mut n = FitzHughNagumoNeuron::new();
        for _ in 0..2000 {
            n.step(2.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn fhn_recovery_variable() {
        let mut n = FitzHughNagumoNeuron::new();
        for _ in 0..2000 {
            n.step(1.0);
        }
        // w should have evolved from initial
        assert!((n.w - (-0.5)).abs() > 0.01, "recovery w should change");
    }

    #[test]
    fn fhn_invalid_input_preserves_state() {
        let mut n = FitzHughNagumoNeuron::new();
        let before = n.clone();
        assert_eq!(n.step(f64::NAN), -1);
        assert_eq!(n.v, before.v);
        assert_eq!(n.w, before.w);
    }

    #[test]
    fn fhn_overflow_candidate_preserves_state() {
        let mut n = FitzHughNagumoNeuron::new();
        n.v = 1.0e103;
        let before = n.clone();
        assert_eq!(n.step(0.0), -1);
        assert_eq!(n.v, before.v);
        assert_eq!(n.w, before.w);
    }

    #[test]
    fn fhn_negative_no_crash() {
        let mut n = FitzHughNagumoNeuron::new();
        for _ in 0..500 {
            n.step(-5.0);
        }
        assert!(n.v.is_finite());
    }
}
