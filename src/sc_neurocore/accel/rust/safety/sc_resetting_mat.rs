// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety oracle for SC resetting MAT

//! Dependency-free preservation of the historical SC RK4/reset recurrence.

const V_MIN: f64 = -200.0;
const V_MAX: f64 = 100.0;
const THETA_MAX: f64 = 1.0e9;

/// Complete SC resetting-MAT state and configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct SCResettingMATNeuron {
    /// Membrane voltage.
    pub v: f64,
    /// Fast threshold state.
    pub theta1: f64,
    /// Slow threshold state.
    pub theta2: f64,
    /// Resting voltage.
    pub v_rest: f64,
    /// Post-event reset voltage.
    pub v_reset: f64,
    /// Baseline threshold.
    pub v_threshold_base: f64,
    /// Membrane time constant.
    pub tau_m: f64,
    /// Fast threshold time constant.
    pub tau_1: f64,
    /// Slow threshold time constant.
    pub tau_2: f64,
    /// Fast event increment.
    pub h1: f64,
    /// Slow event increment.
    pub h2: f64,
    /// Input resistance.
    pub resistance: f64,
    /// RK4 timestep.
    pub dt: f64,
}

impl SCResettingMATNeuron {
    /// Construct the historical SC defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            v: -70.0,
            theta1: 0.0,
            theta2: 0.0,
            v_rest: -70.0,
            v_reset: -70.0,
            v_threshold_base: -50.0,
            tau_m: 10.0,
            tau_1: 10.0,
            tau_2: 200.0,
            h1: 5.0,
            h2: 3.0,
            resistance: 1.0,
            dt: 1.0,
        }
    }

    /// Return whether all invariants hold.
    #[must_use]
    pub fn validate(&self) -> bool {
        [
            self.v,
            self.theta1,
            self.theta2,
            self.v_rest,
            self.v_reset,
            self.v_threshold_base,
            self.tau_m,
            self.tau_1,
            self.tau_2,
            self.h1,
            self.h2,
            self.resistance,
            self.dt,
        ]
        .iter()
        .all(|value| value.is_finite())
            && (V_MIN..=V_MAX).contains(&self.v)
            && (V_MIN..=V_MAX).contains(&self.v_reset)
            && [self.theta1, self.theta2, self.h1, self.h2]
                .iter()
                .all(|value| (0.0..=THETA_MAX).contains(value))
            && self.tau_m > 0.0
            && self.tau_1 > 0.0
            && self.tau_2 > 0.0
            && self.resistance > 0.0
            && self.dt > 0.0
    }

    fn derivatives(&self, v: f64, theta1: f64, theta2: f64, current: f64) -> [f64; 3] {
        [
            (-(v - self.v_rest) + self.resistance * current) / self.tau_m,
            -theta1 / self.tau_1,
            -theta2 / self.tau_2,
        ]
    }

    fn candidate(&self, current: f64) -> [f64; 3] {
        let k1 = self.derivatives(self.v, self.theta1, self.theta2, current);
        let k2 = self.derivatives(
            self.v + 0.5 * self.dt * k1[0],
            self.theta1 + 0.5 * self.dt * k1[1],
            self.theta2 + 0.5 * self.dt * k1[2],
            current,
        );
        let k3 = self.derivatives(
            self.v + 0.5 * self.dt * k2[0],
            self.theta1 + 0.5 * self.dt * k2[1],
            self.theta2 + 0.5 * self.dt * k2[2],
            current,
        );
        let k4 = self.derivatives(
            self.v + self.dt * k3[0],
            self.theta1 + self.dt * k3[1],
            self.theta2 + self.dt * k3[2],
            current,
        );
        let scale = self.dt / 6.0;
        [
            self.v + scale * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
            self.theta1 + scale * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
            self.theta2 + scale * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
        ]
    }

    /// Advance one atomic RK4/reset step; return `-1` on invalid input/state.
    pub fn step(&mut self, current: f64) -> i32 {
        if !current.is_finite() || !self.validate() {
            return -1;
        }
        let [mut v, mut theta1, mut theta2] = self.candidate(current);
        if ![v, theta1, theta2].iter().all(|value| value.is_finite())
            || !(V_MIN..=V_MAX).contains(&v)
            || !(0.0..=THETA_MAX).contains(&theta1)
            || !(0.0..=THETA_MAX).contains(&theta2)
        {
            return -1;
        }
        let spike = v >= self.v_threshold_base + theta1 + theta2;
        if spike {
            theta1 += self.h1;
            theta2 += self.h2;
            if theta1 > THETA_MAX || theta2 > THETA_MAX {
                return -1;
            }
            v = self.v_reset;
        }
        self.v = v;
        self.theta1 = theta1;
        self.theta2 = theta2;
        i32::from(spike)
    }

    /// Reset dynamic state while preserving configuration.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.theta1 = 0.0;
        self.theta2 = 0.0;
    }
}

impl Default for SCResettingMATNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate an SC resetting-MAT state without mutation.
#[must_use]
pub fn validate_sc_resetting_mat(state: &SCResettingMATNeuron) -> bool {
    state.validate()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn historical_anchor_and_atomic_failure() {
        let mut neuron = SCResettingMATNeuron::new();
        let currents = std::iter::repeat_n(0.0, 32)
            .chain(std::iter::repeat_n(50.0, 96))
            .chain((0..128).map(|index| if index % 2 == 0 { 20.0 } else { 60.0 }));
        assert_eq!(currents.map(|current| neuron.step(current)).sum::<i32>(), 13);
        assert_eq!(neuron.theta1, 5.262_135_955_944_077);
        assert_eq!(neuron.theta2, 21.149_478_444_493_045);
        let before = neuron.clone();
        assert_eq!(neuron.step(f64::NAN), -1);
        assert_eq!(neuron, before);
    }
}
