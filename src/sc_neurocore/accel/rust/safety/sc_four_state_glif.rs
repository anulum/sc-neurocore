// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — standalone Rust safety retained four-state GLIF

//! Count-neutral custody for the historical four-state RK4 recurrence.

/// Historical SC-NeuroCore four-state adaptive-threshold recurrence.
#[derive(Clone, Debug)]
pub struct SCFourStateGLIFNeuron {
    pub v: f64,
    pub theta: f64,
    pub theta_inf: f64,
    pub i_asc1: f64,
    pub i_asc2: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub tau_m: f64,
    pub tau_theta: f64,
    pub tau_asc1: f64,
    pub tau_asc2: f64,
    pub a_theta: f64,
    pub delta_theta: f64,
    pub r_asc1: f64,
    pub r_asc2: f64,
    pub resistance: f64,
    pub dt: f64,
}

impl SCFourStateGLIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            theta: -50.0,
            theta_inf: -50.0,
            i_asc1: 0.0,
            i_asc2: 0.0,
            v_rest: -70.0,
            v_reset: -70.0,
            tau_m: 10.0,
            tau_theta: 100.0,
            tau_asc1: 10.0,
            tau_asc2: 200.0,
            a_theta: 0.01,
            delta_theta: 2.0,
            r_asc1: 1.0,
            r_asc2: 0.5,
            resistance: 1.0,
            dt: 1.0,
        }
    }

    fn finite(values: &[f64]) -> bool {
        values.iter().all(|value| value.is_finite())
    }

    fn valid(&self) -> bool {
        Self::finite(&[
            self.v,
            self.theta,
            self.theta_inf,
            self.i_asc1,
            self.i_asc2,
            self.v_rest,
            self.v_reset,
            self.tau_m,
            self.tau_theta,
            self.tau_asc1,
            self.tau_asc2,
            self.a_theta,
            self.delta_theta,
            self.r_asc1,
            self.r_asc2,
            self.resistance,
            self.dt,
        ]) && self.tau_m > 0.0
            && self.tau_theta > 0.0
            && self.tau_asc1 > 0.0
            && self.tau_asc2 > 0.0
            && self.dt > 0.0
            && self.delta_theta >= 0.0
            && self.resistance >= 0.0
    }

    fn derivatives(&self, state: [f64; 4], current: f64) -> [f64; 4] {
        [
            (-(state[0] - self.v_rest) + self.resistance * current + state[2] + state[3])
                / self.tau_m,
            (self.theta_inf - state[1] + self.a_theta * (state[0] - self.v_rest)) / self.tau_theta,
            -state[2] / self.tau_asc1,
            -state[3] / self.tau_asc2,
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
        let state = [self.v, self.theta, self.i_asc1, self.i_asc2];
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
            i_asc1: values[2],
            i_asc2: values[3],
            ..self.clone()
        };
        let event = i32::from(next.v >= next.theta);
        if event == 1 {
            next.v = self.v_reset;
            next.theta += self.delta_theta;
            next.i_asc1 += self.r_asc1;
            next.i_asc2 += self.r_asc2;
        }
        next.valid().then_some((next, event))
    }

    pub fn try_step(&mut self, current: f64) -> Option<i32> {
        let (candidate, event) = self.candidate(current)?;
        *self = candidate;
        Some(event)
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

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
        self.theta = self.theta_inf;
        self.i_asc1 = 0.0;
        self.i_asc2 = 0.0;
    }
}

impl Default for SCFourStateGLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_regimes_are_stable() {
        for (current, expected) in [(0.0, 0), (30.0, 54), (50.0, 95)] {
            let mut neuron = SCFourStateGLIFNeuron::new();
            let (_, events) = neuron
                .try_simulate(1000, current)
                .expect("valid recurrence");
            assert_eq!(events, expected, "current={current}");
        }
    }

    #[test]
    fn invalid_batch_is_failure_atomic() {
        let mut neuron = SCFourStateGLIFNeuron::new();
        let before = neuron.clone();
        assert!(neuron.try_simulate(4, f64::NAN).is_none());
        assert_eq!(neuron.v, before.v);
        assert_eq!(neuron.theta, before.theta);
    }
}
