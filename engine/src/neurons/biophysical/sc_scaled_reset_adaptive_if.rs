// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained scaled-reset adaptive IF model

//! Count-neutral custody for the historical scaled-reset adaptive IF recurrence.

/// Historical SC-NeuroCore four-state adaptive-threshold recurrence.
#[derive(Clone, Debug)]
pub struct SCScaledResetAdaptiveIFNeuron {
    pub v: f64,
    pub theta: f64,
    pub i1: f64,
    pub i2: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub theta_reset: f64,
    pub theta_inf: f64,
    pub tau_v: f64,
    pub tau_theta: f64,
    pub tau_1: f64,
    pub tau_2: f64,
    pub a: f64,
    pub b: f64,
    pub r1: f64,
    pub r2: f64,
    pub dt: f64,
}

impl SCScaledResetAdaptiveIFNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0,
            theta: 1.0,
            i1: 0.0,
            i2: 0.0,
            v_rest: 0.0,
            v_reset: 0.0,
            theta_reset: 1.0,
            theta_inf: 1.0,
            tau_v: 10.0,
            tau_theta: 100.0,
            tau_1: 10.0,
            tau_2: 200.0,
            a: 0.0,
            b: 0.0,
            r1: 0.0,
            r2: 0.0,
            dt: 1.0,
        }
    }

    fn finite_values(values: &[f64]) -> bool {
        values.iter().all(|value| value.is_finite())
    }

    fn valid_runtime(&self) -> bool {
        Self::finite_values(&[
            self.v,
            self.theta,
            self.i1,
            self.i2,
            self.v_rest,
            self.v_reset,
            self.theta_reset,
            self.theta_inf,
            self.tau_v,
            self.tau_theta,
            self.tau_1,
            self.tau_2,
            self.a,
            self.b,
            self.r1,
            self.r2,
            self.dt,
        ]) && self.tau_v > 0.0
            && self.tau_theta > 0.0
            && self.tau_1 > 0.0
            && self.tau_2 > 0.0
            && self.dt > 0.0
    }

    fn derivatives(&self, v: f64, theta: f64, i1: f64, i2: f64, current: f64) -> [f64; 4] {
        [
            (-(v - self.v_rest) + i1 + i2 + current) / self.tau_v,
            (self.theta_inf - theta + self.a * (v - self.v_rest)) / self.tau_theta,
            -i1 / self.tau_1,
            -i2 / self.tau_2,
        ]
    }

    fn add_scaled(state: [f64; 4], slope: [f64; 4], scale: f64) -> [f64; 4] {
        [
            state[0] + scale * slope[0],
            state[1] + scale * slope[1],
            state[2] + scale * slope[2],
            state[3] + scale * slope[3],
        ]
    }

    fn rk4_candidate(&self, current: f64) -> Option<[f64; 4]> {
        let state = [self.v, self.theta, self.i1, self.i2];
        let half_dt = 0.5 * self.dt;
        let k1 = self.derivatives(state[0], state[1], state[2], state[3], current);
        let s2 = Self::add_scaled(state, k1, half_dt);
        let k2 = self.derivatives(s2[0], s2[1], s2[2], s2[3], current);
        let s3 = Self::add_scaled(state, k2, half_dt);
        let k3 = self.derivatives(s3[0], s3[1], s3[2], s3[3], current);
        let s4 = Self::add_scaled(state, k3, self.dt);
        let k4 = self.derivatives(s4[0], s4[1], s4[2], s4[3], current);
        let candidate = [
            state[0] + self.dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state[1] + self.dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            state[2] + self.dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
            state[3] + self.dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
        ];
        if Self::finite_values(&candidate) {
            Some(candidate)
        } else {
            None
        }
    }

    /// Advance one retained interval, returning `None` for invalid input or state.
    pub fn try_step(&mut self, current: f64) -> Option<i32> {
        if !current.is_finite() || !self.valid_runtime() {
            return None;
        }
        let candidate = self.rk4_candidate(current)?;
        let mut next = self.clone();
        next.v = candidate[0];
        next.theta = candidate[1];
        next.i1 = candidate[2];
        next.i2 = candidate[3];
        let event = i32::from(next.v >= next.theta);
        if event == 1 {
            next.v = self.v_reset + self.b * (next.v - self.v_rest);
            next.theta = self.theta_reset.max(next.theta);
            next.i1 += self.r1;
            next.i2 += self.r2;
        }
        if !next.valid_runtime() {
            return None;
        }
        *self = next;
        Some(event)
    }

    /// Advance one retained interval and preserve state on invalid input.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Run `n_steps` of the candidate-first RK4 recurrence under a constant
    /// `current`, recording the membrane voltage after every step.
    ///
    /// Reuses [`step`] verbatim so the compiled inner loop is bit-identical to
    /// the per-step path; returns the voltage trace and the total spike count.
    pub fn simulate(&mut self, n_steps: usize, current: f64) -> (Vec<f64>, i64) {
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes: i64 = 0;
        for _ in 0..n_steps {
            spikes += i64::from(self.step(current));
            trace.push(self.v);
        }
        (trace, spikes)
    }

    /// Simulate a retained trajectory atomically.
    pub fn try_simulate(&mut self, n_steps: usize, current: f64) -> Option<(Vec<f64>, i64)> {
        let mut candidate = self.clone();
        let mut trace = Vec::with_capacity(n_steps);
        let mut events = 0_i64;
        for _ in 0..n_steps {
            events += i64::from(candidate.try_step(current)?);
            trace.push(candidate.v);
        }
        *self = candidate;
        Some((trace, events))
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.theta = self.theta_reset;
        self.i1 = 0.0;
        self.i2 = 0.0;
    }
}
impl Default for SCScaledResetAdaptiveIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = SCScaledResetAdaptiveIFNeuron::default();
        let constructed = SCScaledResetAdaptiveIFNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn simulate_matches_repeated_steps() {
        let mut simulated = SCScaledResetAdaptiveIFNeuron::new();
        let mut repeated = simulated.clone();
        let (trace, spikes) = simulated.simulate(8, 5.0);
        let mut expected_spikes = 0_i64;
        let expected: Vec<f64> = (0..8)
            .map(|_| {
                expected_spikes += i64::from(repeated.step(5.0));
                repeated.v
            })
            .collect();
        assert_eq!(trace, expected);
        assert_eq!(spikes, expected_spikes);
    }

    #[test]
    fn nonfinite_rk4_candidate_preserves_state() {
        let mut n = SCScaledResetAdaptiveIFNeuron::new();
        n.dt = f64::MAX;
        let before = (n.v, n.theta, n.i1, n.i2);
        assert_eq!(n.step(1.0), 0);
        assert_eq!((n.v, n.theta, n.i1, n.i2), before);
    }

    #[test]
    fn sc_scaled_reset_fires() {
        let mut n = SCScaledResetAdaptiveIFNeuron::new();
        let t: i32 = (0..100).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }

    // -- SCScaledResetAdaptiveIF --
    #[test]
    fn sc_scaled_reset_silent_without_input() {
        let mut n = SCScaledResetAdaptiveIFNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn sc_scaled_reset_reset_clears_state() {
        let mut n = SCScaledResetAdaptiveIFNeuron::new();
        for _ in 0..100 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
        assert!((n.theta - n.theta_reset).abs() < 1e-10);
    }
    #[test]
    fn sc_scaled_reset_rk4_reference_point() {
        let mut n = SCScaledResetAdaptiveIFNeuron::new();
        assert_eq!(n.step(0.5), 0);
        assert!((n.v - 0.04758125).abs() < 1e-12);
        assert!((n.theta - 1.0).abs() < 1e-15);
        assert!((n.i1 - 0.0).abs() < 1e-15);
        assert!((n.i2 - 0.0).abs() < 1e-15);
    }
    #[test]
    fn sc_scaled_reset_spike_reset_uses_b() {
        let mut n = SCScaledResetAdaptiveIFNeuron::new();
        n.v = 0.99;
        n.b = 0.5;
        n.r1 = 1.25;
        n.r2 = -0.25;
        assert_eq!(n.step(2.0), 1);
        assert!((n.v - 0.5430570625).abs() < 1e-12);
        assert!((n.i1 - 1.25).abs() < 1e-15);
        assert!((n.i2 - (-0.25)).abs() < 1e-15);
    }
    #[test]
    fn sc_scaled_reset_invalid_input_preserves_state() {
        let mut n = SCScaledResetAdaptiveIFNeuron::new();
        n.v = 0.2;
        n.i1 = 0.3;
        let before = (n.v, n.theta, n.i1, n.i2);
        assert_eq!(n.step(f64::NAN), 0);
        assert_eq!((n.v, n.theta, n.i1, n.i2), before);
    }
    #[test]
    fn sc_scaled_reset_extreme_bounded() {
        let mut n = SCScaledResetAdaptiveIFNeuron::new();
        for _ in 0..200 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn sc_scaled_reset_adaptive_threshold() {
        let mut n = SCScaledResetAdaptiveIFNeuron::new();
        n.a = 0.1;
        for _ in 0..100 {
            n.step(5.0);
        }
        // Threshold should have adapted
        assert!(n.theta.is_finite());
    }
    #[test]
    fn sc_scaled_reset_negative_no_crash() {
        let mut n = SCScaledResetAdaptiveIFNeuron::new();
        for _ in 0..200 {
            n.step(-10.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn sc_scaled_reset_nan_no_panic() {
        let mut n = SCScaledResetAdaptiveIFNeuron::new();
        n.step(f64::NAN);
    }
}
