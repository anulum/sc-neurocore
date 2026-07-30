// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Retained triangular piecewise-linear oscillator

//! SCTriangularMcKean piecewise-linear neuron dynamics.

/// Retained SC triangular piecewise-linear FHN-like recurrence.
#[derive(Clone, Debug)]
pub struct SCTriangularMcKeanNeuron {
    pub v: f64,
    pub w: f64,
    pub a: f64,
    pub epsilon: f64,
    pub gamma: f64,
    pub dt: f64,
    pub v_peak: f64,
}

impl SCTriangularMcKeanNeuron {
    /// Construct the project recurrence with its compatibility defaults.
    pub fn new() -> Self {
        Self {
            v: 0.0,
            w: 0.0,
            a: 0.25,
            epsilon: 0.01,
            gamma: 0.5,
            dt: 0.1,
            v_peak: 0.8,
        }
    }
    fn f_v(&self, v: f64) -> f64 {
        let half_a = self.a / 2.0;
        let mid = (1.0 + self.a) / 2.0;
        if v < half_a {
            -v
        } else if v < mid {
            v - self.a
        } else {
            1.0 - v
        }
    }
    fn valid_numeric_contract(&self) -> bool {
        self.v.is_finite()
            && self.w.is_finite()
            && self.a.is_finite()
            && self.epsilon.is_finite()
            && self.gamma.is_finite()
            && self.dt.is_finite()
            && self.v_peak.is_finite()
            && self.a > 0.0
            && self.a < 1.0
            && self.epsilon > 0.0
            && self.gamma > 0.0
            && self.dt > 0.0
    }
    fn derivatives(&self, v: f64, w: f64, current: f64) -> Option<(f64, f64)> {
        if !(v.is_finite() && w.is_finite() && current.is_finite()) {
            return None;
        }
        let dv = self.f_v(v) - w + current;
        let dw = self.epsilon * (v - self.gamma * w);
        if dv.is_finite() && dw.is_finite() {
            Some((dv, dw))
        } else {
            None
        }
    }
    fn rk4_candidate(&self, current: f64) -> Option<(f64, f64)> {
        let v0 = self.v;
        let w0 = self.w;
        let dt = self.dt;
        let k1 = self.derivatives(v0, w0, current)?;
        let k2 = self.derivatives(v0 + 0.5 * dt * k1.0, w0 + 0.5 * dt * k1.1, current)?;
        let k3 = self.derivatives(v0 + 0.5 * dt * k2.0, w0 + 0.5 * dt * k2.1, current)?;
        let k4 = self.derivatives(v0 + dt * k3.0, w0 + dt * k3.1, current)?;
        let next_v = v0 + dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0;
        let next_w = w0 + dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0;
        if next_v.is_finite() && next_w.is_finite() {
            Some((next_v, next_w))
        } else {
            None
        }
    }
    /// Advance one RK4 sample and report an upward `v_peak` crossing.
    pub fn step(&mut self, current: f64) -> i32 {
        if !self.valid_numeric_contract() || !current.is_finite() {
            return 0;
        }
        let v_prev = self.v;
        let (next_v, next_w) = match self.rk4_candidate(current) {
            Some(candidate) => candidate,
            None => return 0,
        };
        self.v = next_v;
        self.w = next_w;
        if self.v >= self.v_peak && v_prev < self.v_peak {
            1
        } else {
            0
        }
    }
    /// Run `n_steps` RK4 updates under a constant input, returning the `v` trace
    /// and the upward-`v_peak`-crossing spike count. Reuses `step`, so the trace
    /// is bit-identical to the per-step path and to the Python reference (the
    /// piecewise-linear RHS is exact arithmetic — no transcendental functions).
    /// The final state is left in `self.v` / `self.w`.
    pub fn simulate(&mut self, n_steps: usize, current: f64) -> (Vec<f64>, i64) {
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes: i64 = 0;
        for _ in 0..n_steps {
            let spiked = self.step(current);
            trace.push(self.v);
            spikes += spiked as i64;
        }
        (trace, spikes)
    }
    /// Restore the project recurrence equilibrium state.
    pub fn reset(&mut self) {
        self.v = 0.0;
        self.w = 0.0;
    }
}
impl Default for SCTriangularMcKeanNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = SCTriangularMcKeanNeuron::default();
        let constructed = SCTriangularMcKeanNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn simulate_matches_repeated_step() {
        let mut simulated = SCTriangularMcKeanNeuron::new();
        let mut repeated = SCTriangularMcKeanNeuron::new();
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
    fn sc_triangular_mckean_fires() {
        let mut n = SCTriangularMcKeanNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(0.5)).sum();
        assert!(t > 0);
    }

    #[test]
    fn sc_triangular_mckean_reset_clears_state() {
        let mut n = SCTriangularMcKeanNeuron::new();
        for _ in 0..500 {
            n.step(0.5);
        }
        n.reset();
        assert!((n.v - 0.0).abs() < 1e-10);
    }

    #[test]
    fn sc_triangular_mckean_bounded() {
        let mut n = SCTriangularMcKeanNeuron::new();
        for _ in 0..2000 {
            n.step(50.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn sc_triangular_mckean_matches_rk4_candidate() {
        fn f(v: f64, a: f64) -> f64 {
            let half_a = a / 2.0;
            let mid = (1.0 + a) / 2.0;
            if v < half_a {
                -v
            } else if v < mid {
                v - a
            } else {
                1.0 - v
            }
        }
        fn rhs(n: &SCTriangularMcKeanNeuron, v: f64, w: f64, current: f64) -> (f64, f64) {
            (f(v, n.a) - w + current, n.epsilon * (v - n.gamma * w))
        }

        let mut n = SCTriangularMcKeanNeuron {
            v: 0.2,
            w: -0.1,
            ..Default::default()
        };
        let current = 0.5;
        let v0 = n.v;
        let w0 = n.w;
        let dt = n.dt;
        let k1 = rhs(&n, v0, w0, current);
        let k2 = rhs(&n, v0 + 0.5 * dt * k1.0, w0 + 0.5 * dt * k1.1, current);
        let k3 = rhs(&n, v0 + 0.5 * dt * k2.0, w0 + 0.5 * dt * k2.1, current);
        let k4 = rhs(&n, v0 + dt * k3.0, w0 + dt * k3.1, current);
        let expected_v = v0 + dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0;
        let expected_w = w0 + dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0;

        assert_eq!(n.step(current), 0);
        assert!((n.v - expected_v).abs() < 1e-14);
        assert!((n.w - expected_w).abs() < 1e-14);
    }

    #[test]
    fn sc_triangular_mckean_nan_no_panic() {
        let mut n = SCTriangularMcKeanNeuron::new();
        let before = (n.v, n.w);
        assert_eq!(n.step(f64::NAN), 0);
        assert_eq!((n.v, n.w), before);
    }

    #[test]
    fn sc_triangular_mckean_overflow_candidate_preserves_state() {
        let mut n = SCTriangularMcKeanNeuron {
            v: 1.0e308,
            w: -1.7e308,
            ..Default::default()
        };
        let before = (n.v, n.w);
        assert_eq!(n.step(1.7e308), 0);
        assert_eq!((n.v, n.w), before);
    }

    #[test]
    fn sc_triangular_mckean_negative_no_crash() {
        let mut n = SCTriangularMcKeanNeuron::new();
        for _ in 0..500 {
            n.step(-5.0);
        }
        assert!(n.v.is_finite());
    }
}
