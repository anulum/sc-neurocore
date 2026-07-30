// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for retained normalized energy LIF

//! Fail-closed retained normalized energy-gated exact-flow LIF.

const V_MIN: f64 = -200.0;
const V_MAX: f64 = 100.0;

/// Complete retained SC normalized energy-LIF state.
#[derive(Debug, Clone)]
pub struct SCNormalizedEnergyLIFNeuron {
    pub v: f64,
    pub epsilon: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_e: f64,
    pub alpha: f64,
    pub epsilon_0: f64,
    pub resistance: f64,
    pub dt: f64,
}

impl Default for SCNormalizedEnergyLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl SCNormalizedEnergyLIFNeuron {
    /// Construct the frozen project defaults.
    pub fn new() -> Self {
        Self {
            v: -70.0,
            epsilon: 1.0,
            v_rest: -70.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau_m: 10.0,
            tau_e: 500.0,
            alpha: 0.1,
            epsilon_0: 1.0,
            resistance: 1.0,
            dt: 1.0,
        }
    }

    /// Advance atomically, returning `-1` for invalid input or state.
    pub fn step(&mut self, current: f64) -> i32 {
        if !self.valid() || !current.is_finite() {
            return -1;
        }
        let md = (-self.dt / self.tau_m).exp();
        let ed = (-self.dt / self.tau_e).exp();
        let de = self.epsilon - self.epsilon_0;
        let epsilon = self.epsilon_0 + de * ed;
        let steady = self.epsilon_0 * self.tau_m * (1.0 - md);
        let rate = (1.0 / self.tau_m) - (1.0 / self.tau_e);
        let transient = if rate.abs() < 1.0e-12 {
            de * md * self.dt
        } else {
            de * md * (rate * self.dt).exp_m1() / rate
        };
        let v = self.v_rest
            + (self.v - self.v_rest) * md
            + (self.resistance * current / self.tau_m) * (steady + transient);
        if !(v.is_finite()
            && (V_MIN..=V_MAX).contains(&v)
            && epsilon.is_finite()
            && (0.0..=self.epsilon_0).contains(&epsilon))
        {
            return -1;
        }
        if v >= self.v_threshold && epsilon > 0.1 {
            self.v = self.v_reset;
            self.epsilon = (epsilon - self.alpha).max(0.0);
            return 1;
        }
        self.v = v;
        self.epsilon = epsilon;
        0
    }

    /// Return whether the complete state satisfies the frozen contract.
    pub fn valid(&self) -> bool {
        self.v.is_finite()
            && (V_MIN..=V_MAX).contains(&self.v)
            && self.epsilon.is_finite()
            && self.epsilon_0.is_finite()
            && self.epsilon_0 >= 0.0
            && (0.0..=self.epsilon_0).contains(&self.epsilon)
            && self.v_rest.is_finite()
            && self.v_reset.is_finite()
            && (V_MIN..=V_MAX).contains(&self.v_reset)
            && self.v_threshold.is_finite()
            && self.tau_m.is_finite()
            && self.tau_m > 0.0
            && self.tau_e.is_finite()
            && self.tau_e > 0.0
            && self.alpha.is_finite()
            && self.alpha >= 0.0
            && self.resistance.is_finite()
            && self.resistance > 0.0
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.dt <= self.tau_m
            && self.dt <= self.tau_e
            && self.v_threshold > self.v_rest
            && self.v_threshold > self.v_reset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_transition_is_atomic() {
        let mut state = SCNormalizedEnergyLIFNeuron::new();
        assert_eq!(state.step(30.0), 0);
        let before = state.clone();
        assert_eq!(state.step(f64::NAN), -1);
        assert_eq!((state.v, state.epsilon), (before.v, before.epsilon));
    }
}
