// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety oracle for source MAT*

//! Dependency-free safety implementation of the non-resetting MAT* equations.

const V_MIN: f64 = -200.0;
const V_MAX: f64 = 200.0;
const THETA_MAX: f64 = 1.0e9;

/// Complete source MAT* state and paper-profile configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct MATNeuron {
    /// Relative membrane voltage in mV.
    pub v: f64,
    /// Fast threshold-history contribution in mV.
    pub theta1: f64,
    /// Slow threshold-history contribution in mV.
    pub theta2: f64,
    /// Remaining absolute refractory interval in ms.
    pub refractory_remaining: f64,
    /// Baseline threshold in mV.
    pub omega: f64,
    /// Membrane time constant in ms.
    pub tau_m: f64,
    /// Fast history time constant in ms.
    pub tau_1: f64,
    /// Slow history time constant in ms.
    pub tau_2: f64,
    /// Fast post-event threshold increment in mV.
    pub alpha_1: f64,
    /// Slow post-event threshold increment in mV.
    pub alpha_2: f64,
    /// Input resistance in megaohms.
    pub resistance: f64,
    /// Absolute refractory duration in ms.
    pub refractory_period: f64,
    /// Forward-Euler timestep in ms.
    pub dt: f64,
}

impl MATNeuron {
    /// Construct the paper's regular-spiking example profile.
    #[must_use]
    pub fn new() -> Self {
        Self {
            v: 0.0,
            theta1: 0.0,
            theta2: 0.0,
            refractory_remaining: 0.0,
            omega: 19.0,
            tau_m: 5.0,
            tau_1: 10.0,
            tau_2: 200.0,
            alpha_1: 37.0,
            alpha_2: 2.0,
            resistance: 50.0,
            refractory_period: 2.0,
            dt: 0.001,
        }
    }

    /// Return whether all invariants hold.
    #[must_use]
    pub fn validate(&self) -> bool {
        [
            self.v,
            self.theta1,
            self.theta2,
            self.refractory_remaining,
            self.omega,
            self.tau_m,
            self.tau_1,
            self.tau_2,
            self.alpha_1,
            self.alpha_2,
            self.resistance,
            self.refractory_period,
            self.dt,
        ]
        .iter()
        .all(|value| value.is_finite())
            && (V_MIN..=V_MAX).contains(&self.v)
            && (-THETA_MAX..=THETA_MAX).contains(&self.omega)
            && [self.theta1, self.theta2, self.alpha_1, self.alpha_2]
                .iter()
                .all(|value| (0.0..=THETA_MAX).contains(value))
            && self.tau_m > 0.0
            && self.tau_1 > 0.0
            && self.tau_2 > 0.0
            && self.resistance > 0.0
            && self.refractory_period >= 0.0
            && self.dt > 0.0
            && (0.0..=self.refractory_period).contains(&self.refractory_remaining)
    }

    /// Advance one atomic MAT* step; return `-1` on invalid input/state.
    pub fn step(&mut self, current: f64) -> i32 {
        if !current.is_finite() || !self.validate() {
            return -1;
        }
        let v = self.v + self.dt * (-self.v + self.resistance * current) / self.tau_m;
        let mut theta1 = self.theta1 * (-self.dt / self.tau_1).exp();
        let mut theta2 = self.theta2 * (-self.dt / self.tau_2).exp();
        let mut refractory = (self.refractory_remaining - self.dt).max(0.0);
        if ![v, theta1, theta2, refractory]
            .iter()
            .all(|value| value.is_finite())
            || !(V_MIN..=V_MAX).contains(&v)
            || !(0.0..=THETA_MAX).contains(&theta1)
            || !(0.0..=THETA_MAX).contains(&theta2)
        {
            return -1;
        }
        let spike = refractory == 0.0 && v >= self.omega + theta1 + theta2;
        if spike {
            theta1 += self.alpha_1;
            theta2 += self.alpha_2;
            refractory = self.refractory_period;
            if theta1 > THETA_MAX || theta2 > THETA_MAX {
                return -1;
            }
        }
        self.v = v;
        self.theta1 = theta1;
        self.theta2 = theta2;
        self.refractory_remaining = refractory;
        i32::from(spike)
    }

    /// Reset all dynamic state while retaining configuration.
    pub fn reset(&mut self) {
        self.v = 0.0;
        self.theta1 = 0.0;
        self.theta2 = 0.0;
        self.refractory_remaining = 0.0;
    }
}

impl Default for MATNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate a source MAT* state without mutation.
#[must_use]
pub fn validate_mat(state: &MATNeuron) -> bool {
    state.validate()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_resetting_event_and_atomic_failure() {
        let mut neuron = MATNeuron {
            v: 20.0,
            ..MATNeuron::new()
        };
        assert_eq!(neuron.step(0.0), 1);
        assert!(neuron.v > 19.0);
        let before = neuron.clone();
        assert_eq!(neuron.step(f64::NAN), -1);
        assert_eq!(neuron, before);
    }
}
