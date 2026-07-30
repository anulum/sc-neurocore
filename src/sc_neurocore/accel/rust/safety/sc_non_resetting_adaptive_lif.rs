// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! Fail-closed Rust safety implementation of the retained SC recurrence.

/// Complete historical SC exact-relaxation state and configuration.
#[derive(Debug, Clone)]
pub struct SCNonResettingAdaptiveLIFNeuron {
    /// Non-resetting membrane voltage, in mV.
    pub v: f64,
    /// Adaptive threshold, in mV.
    pub theta: f64,
    /// Membrane rest, in mV.
    pub v_rest: f64,
    /// Threshold rest, in mV.
    pub theta_rest: f64,
    /// Threshold increment per event, in mV.
    pub delta_theta: f64,
    /// Membrane time constant, in ms.
    pub tau_m: f64,
    /// Threshold time constant, in ms.
    pub tau_theta: f64,
    /// Current-to-voltage gain.
    pub r_m: f64,
    /// Sample interval, in ms.
    pub dt: f64,
}

impl SCNonResettingAdaptiveLIFNeuron {
    /// Construct the frozen project defaults.
    pub fn new() -> Self {
        Self {
            v: -65.0,
            theta: -50.0,
            v_rest: -65.0,
            theta_rest: -50.0,
            delta_theta: 5.0,
            tau_m: 10.0,
            tau_theta: 50.0,
            r_m: 1.0,
            dt: 0.1,
        }
    }
    /// Advance one atomic exact-relaxation sample.
    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() || !self.valid() {
            return Err("invalid SC adaptive LIF state, configuration, or current");
        }
        let steady = self.v_rest + self.r_m * current;
        let dv = (-self.dt / self.tau_m).exp();
        let dt = (-self.dt / self.tau_theta).exp();
        let v = dv * self.v + (1.0 - dv) * steady;
        let mut theta = dt * self.theta + (1.0 - dt) * self.theta_rest;
        if !steady.is_finite() || !v.is_finite() || !theta.is_finite() {
            return Err("SC adaptive LIF candidate is non-finite");
        }
        let spike = v >= theta;
        if spike {
            theta += self.delta_theta;
        }
        if !theta.is_finite() {
            return Err("SC adaptive LIF threshold is non-finite");
        }
        self.v = v;
        self.theta = theta;
        Ok(i32::from(spike))
    }
    /// Return whether all project invariants hold.
    pub fn valid(&self) -> bool {
        [
            self.v,
            self.theta,
            self.v_rest,
            self.theta_rest,
            self.delta_theta,
            self.tau_m,
            self.tau_theta,
            self.r_m,
            self.dt,
        ]
        .iter()
        .all(|v| v.is_finite())
            && self.delta_theta >= 0.0
            && self.r_m >= 0.0
            && self.tau_m > 0.0
            && self.tau_theta > 0.0
            && self.dt > 0.0
    }
    /// Restore configured rest state.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.theta = self.theta_rest;
    }
}

impl Default for SCNonResettingAdaptiveLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_transition_is_atomic() {
        let mut n = SCNonResettingAdaptiveLIFNeuron::new();
        let before = (n.v, n.theta);
        assert!(n.step(f64::NAN).is_err());
        assert_eq!((n.v, n.theta), before);
    }
}
