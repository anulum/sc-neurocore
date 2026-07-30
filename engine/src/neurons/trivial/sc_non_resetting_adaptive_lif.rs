// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! Retained SC exact-relaxation adaptive LIF recurrence.

/// Historical project state and configuration formerly named NonResettingLIF.
#[derive(Clone, Debug)]
pub struct SCNonResettingAdaptiveLIFNeuron {
    /// Non-resetting membrane voltage, in mV.
    pub v: f64,
    /// Adaptive event threshold, in mV.
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
    /// Project current-to-voltage gain.
    pub r_m: f64,
    /// Sample interval, in ms.
    pub dt: f64,
}

impl SCNonResettingAdaptiveLIFNeuron {
    /// Construct the frozen historical project profile.
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

    /// Validate the complete project recurrence.
    pub fn validate(&self) -> bool {
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
        .all(|value| value.is_finite())
            && self.delta_theta >= 0.0
            && self.r_m >= 0.0
            && self.tau_m > 0.0
            && self.tau_theta > 0.0
            && self.dt > 0.0
    }

    /// Advance one atomic exact-relaxation sample.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() || !self.validate() {
            return Err("invalid SC non-resetting adaptive LIF state, configuration, or current");
        }
        let steady = self.v_rest + self.r_m * current;
        if !steady.is_finite() {
            return Err("SC non-resetting adaptive LIF steady state is non-finite");
        }
        let v_decay = (-self.dt / self.tau_m).exp();
        let theta_decay = (-self.dt / self.tau_theta).exp();
        let v = v_decay * self.v + (1.0 - v_decay) * steady;
        let mut theta = theta_decay * self.theta + (1.0 - theta_decay) * self.theta_rest;
        if !v.is_finite() || !theta.is_finite() {
            return Err("SC non-resetting adaptive LIF candidate is non-finite");
        }
        let spike = v >= theta;
        if spike {
            theta += self.delta_theta;
        }
        if !theta.is_finite() {
            return Err("SC non-resetting adaptive LIF threshold is non-finite");
        }
        self.v = v;
        self.theta = theta;
        Ok(i32::from(spike))
    }

    /// Advance one engine-adapter step; invalid state returns no event.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Restore voltage and threshold to configured rests.
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
    fn exact_relaxation_and_atomic_failure() {
        let mut neuron = SCNonResettingAdaptiveLIFNeuron::new();
        let before = (neuron.v, neuron.theta);
        assert!(neuron.try_step(f64::NAN).is_err());
        assert_eq!((neuron.v, neuron.theta), before);
        assert_eq!(neuron.try_step(20.0), Ok(0));
        assert!(neuron.v > -65.0);
    }
}
