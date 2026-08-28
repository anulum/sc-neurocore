// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — source-faithful Mihalas-Niebur generalised IF model

//! Mihalaş-Niebur equations 2.1–2.2 with capacitance-normalised currents.

/// Four-state generalised integrate-and-fire neuron from Mihalaş and Niebur (2009).
///
/// Rates are per millisecond, voltages are volts, and currents are volts per
/// millisecond after division by capacitance. The flow uses fixed-grid RK4 and
/// sampled threshold detection; the published differential equations and event
/// reset are otherwise unchanged.
#[derive(Clone, Debug)]
pub struct MihalasNieburNeuron {
    pub v: f64,
    pub theta: f64,
    pub i1: f64,
    pub i2: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub theta_reset: f64,
    pub theta_inf: f64,
    pub leak_rate: f64,
    pub threshold_voltage_coupling: f64,
    pub threshold_decay_rate: f64,
    pub current_decay_rate_1: f64,
    pub current_decay_rate_2: f64,
    pub current_retention_1: f64,
    pub current_retention_2: f64,
    pub current_jump_1: f64,
    pub current_jump_2: f64,
    pub dt: f64,
}

impl MihalasNieburNeuron {
    /// Construct the paper's common Table 1 profile with Figure 1C coupling.
    pub fn new() -> Self {
        Self {
            v: -0.07,
            theta: -0.05,
            i1: 0.0,
            i2: 0.0,
            v_rest: -0.07,
            v_reset: -0.07,
            theta_reset: -0.06,
            theta_inf: -0.05,
            leak_rate: 0.05,
            threshold_voltage_coupling: 0.005,
            threshold_decay_rate: 0.01,
            current_decay_rate_1: 0.2,
            current_decay_rate_2: 0.02,
            current_retention_1: 0.0,
            current_retention_2: 1.0,
            current_jump_1: 0.0,
            current_jump_2: 0.0,
            dt: 0.1,
        }
    }

    fn finite(values: &[f64]) -> bool {
        values.iter().all(|value| value.is_finite())
    }

    fn valid(&self) -> bool {
        Self::finite(&[
            self.v,
            self.theta,
            self.i1,
            self.i2,
            self.v_rest,
            self.v_reset,
            self.theta_reset,
            self.theta_inf,
            self.leak_rate,
            self.threshold_voltage_coupling,
            self.threshold_decay_rate,
            self.current_decay_rate_1,
            self.current_decay_rate_2,
            self.current_retention_1,
            self.current_retention_2,
            self.current_jump_1,
            self.current_jump_2,
            self.dt,
        ]) && self.leak_rate > 0.0
            && self.threshold_decay_rate > 0.0
            && self.current_decay_rate_1 > 0.0
            && self.current_decay_rate_2 > 0.0
            && self.dt > 0.0
            && self.theta_reset > self.v_reset
    }

    fn derivatives(&self, state: [f64; 4], current: f64) -> [f64; 4] {
        [
            current + state[2] + state[3] - self.leak_rate * (state[0] - self.v_rest),
            self.threshold_voltage_coupling * (state[0] - self.v_rest)
                - self.threshold_decay_rate * (state[1] - self.theta_inf),
            -self.current_decay_rate_1 * state[2],
            -self.current_decay_rate_2 * state[3],
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

    fn candidate(&self, current: f64) -> Option<(Self, i32)> {
        if !self.valid() || !current.is_finite() {
            return None;
        }
        let state = [self.v, self.theta, self.i1, self.i2];
        let half_dt = 0.5 * self.dt;
        let k1 = self.derivatives(state, current);
        let k2 = self.derivatives(Self::add_scaled(state, k1, half_dt), current);
        let k3 = self.derivatives(Self::add_scaled(state, k2, half_dt), current);
        let k4 = self.derivatives(Self::add_scaled(state, k3, self.dt), current);
        let values = [
            state[0] + self.dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state[1] + self.dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            state[2] + self.dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
            state[3] + self.dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
        ];
        if !Self::finite(&values) {
            return None;
        }
        let mut next = Self {
            v: values[0],
            theta: values[1],
            i1: values[2],
            i2: values[3],
            ..self.clone()
        };
        let event = i32::from(next.v >= next.theta);
        if event == 1 {
            next.i1 = self.current_retention_1 * next.i1 + self.current_jump_1;
            next.i2 = self.current_retention_2 * next.i2 + self.current_jump_2;
            next.v = self.v_reset;
            next.theta = self.theta_reset.max(next.theta);
        }
        next.valid().then_some((next, event))
    }

    /// Advance one sampled interval, returning `None` for an invalid candidate.
    pub fn try_step(&mut self, current: f64) -> Option<i32> {
        let (candidate, event) = self.candidate(current)?;
        *self = candidate;
        Some(event)
    }

    /// Advance one sampled interval and leave state unchanged on invalid input.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Simulate a constant-current trajectory atomically.
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

    /// Restore the paper-profile resting state.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.theta = self.theta_inf;
        self.i1 = 0.0;
        self.i2 = 0.0;
    }
}

impl Default for MihalasNieburNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paper_profile_adapts_under_constant_drive() {
        let mut neuron = MihalasNieburNeuron::new();
        let (_, events) = neuron
            .try_simulate(2500, 0.002)
            .expect("valid source profile");
        assert_eq!(events, 13);
        assert!(neuron.theta > neuron.theta_inf);
    }

    #[test]
    fn event_uses_published_reset_map() {
        let mut neuron = MihalasNieburNeuron::new();
        neuron.v = -0.049;
        neuron.i1 = 0.003;
        neuron.i2 = -0.001;
        neuron.current_retention_1 = 0.25;
        neuron.current_retention_2 = 0.5;
        neuron.current_jump_1 = 0.004;
        neuron.current_jump_2 = 0.002;
        assert_eq!(neuron.step(0.02), 1);
        assert_eq!(neuron.v, neuron.v_reset);
        assert!(neuron.theta >= neuron.theta_reset);
        assert!(neuron.i1 > neuron.current_jump_1);
        assert!(neuron.i2 > 0.0);
    }

    #[test]
    fn invalid_batch_is_failure_atomic() {
        let mut neuron = MihalasNieburNeuron::new();
        let before = neuron.clone();
        assert!(neuron.try_simulate(4, f64::NAN).is_none());
        assert_eq!(neuron.v, before.v);
        assert_eq!(neuron.theta, before.theta);
    }
}
