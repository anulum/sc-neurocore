// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! Source-faithful Kobayashi MAT(1) non-resetting neuron.

const V_MIN: f64 = -200.0;
const V_MAX: f64 = 200.0;
const THETA_MAX: f64 = 1.0e9;

/// One-timescale MAT(1) state, configuration, and atomic transition.
#[derive(Clone, Debug)]
pub struct NonResettingLIFNeuron {
    /// Non-resetting membrane voltage relative to rest, in mV.
    pub v: f64,
    /// Single spike-history contribution to the adaptive threshold, in mV.
    pub theta: f64,
    /// Absolute refractory time remaining, in ms.
    pub refractory_remaining: f64,
    /// Resting threshold, in mV.
    pub omega: f64,
    /// Membrane time constant, in ms.
    pub tau_m: f64,
    /// MAT(1) threshold-history time constant, in ms.
    pub tau_theta: f64,
    /// Threshold-history increment per event, in mV.
    pub alpha: f64,
    /// Membrane resistance, in MOhm.
    pub resistance: f64,
    /// Absolute refractory duration, in ms.
    pub refractory_period: f64,
    /// Numerical sample interval, in ms.
    pub dt: f64,
}

impl NonResettingLIFNeuron {
    /// Construct the documented source-equation specialization.
    pub fn new() -> Self {
        Self {
            v: 0.0,
            theta: 0.0,
            refractory_remaining: 0.0,
            omega: 19.0,
            tau_m: 5.0,
            tau_theta: 50.0,
            alpha: 37.0,
            resistance: 50.0,
            refractory_period: 2.0,
            dt: 0.001,
        }
    }

    /// Return the instantaneous adaptive threshold, in mV.
    pub fn threshold(&self) -> f64 {
        self.omega + self.theta
    }

    /// Validate complete dynamic state and configuration.
    pub fn validate(&self) -> bool {
        [
            self.v,
            self.theta,
            self.refractory_remaining,
            self.omega,
            self.tau_m,
            self.tau_theta,
            self.alpha,
            self.resistance,
            self.refractory_period,
            self.dt,
        ]
        .iter()
        .all(|value| value.is_finite())
            && (V_MIN..=V_MAX).contains(&self.v)
            && (0.0..=THETA_MAX).contains(&self.theta)
            && (-THETA_MAX..=THETA_MAX).contains(&self.omega)
            && (0.0..=THETA_MAX).contains(&self.alpha)
            && self.tau_m > 0.0
            && self.tau_theta > 0.0
            && self.resistance > 0.0
            && self.refractory_period >= 0.0
            && self.dt > 0.0
            && (0.0..=self.refractory_period).contains(&self.refractory_remaining)
    }

    /// Advance one atomic MAT(1) sample.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() || !self.validate() {
            return Err("invalid NonResettingLIF state, configuration, or current");
        }
        let v = self.v + self.dt * (-self.v + self.resistance * current) / self.tau_m;
        let mut theta = self.theta * (-self.dt / self.tau_theta).exp();
        let mut refractory = (self.refractory_remaining - self.dt).max(0.0);
        if ![v, theta, refractory].iter().all(|value| value.is_finite())
            || !(V_MIN..=V_MAX).contains(&v)
            || !(0.0..=THETA_MAX).contains(&theta)
        {
            return Err("NonResettingLIF candidate outside safety envelope");
        }
        let spike = refractory == 0.0 && v >= self.omega + theta;
        if spike {
            theta += self.alpha;
            refractory = self.refractory_period;
            if theta > THETA_MAX {
                return Err("NonResettingLIF post-spike threshold outside safety envelope");
            }
        }
        self.v = v;
        self.theta = theta;
        self.refractory_remaining = refractory;
        Ok(i32::from(spike))
    }

    /// Advance one engine-adapter step; invalid state returns no event.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Reset dynamic state while retaining configuration.
    pub fn reset(&mut self) {
        self.v = 0.0;
        self.theta = 0.0;
        self.refractory_remaining = 0.0;
    }
}

impl Default for NonResettingLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_step_is_non_resetting_and_refractory() {
        let mut neuron = NonResettingLIFNeuron {
            v: 20.0,
            ..NonResettingLIFNeuron::new()
        };
        let expected_v = 20.0 + neuron.dt * -20.0 / neuron.tau_m;
        assert_eq!(neuron.try_step(0.0), Ok(1));
        assert_eq!(neuron.v, expected_v);
        assert_eq!(neuron.theta, 37.0);
        assert_eq!(neuron.refractory_remaining, 2.0);
        assert_eq!(neuron.try_step(0.0), Ok(0));
    }

    #[test]
    fn invalid_input_is_atomic() {
        let mut neuron = NonResettingLIFNeuron::new();
        let before = (neuron.v, neuron.theta, neuron.refractory_remaining);
        assert!(neuron.try_step(f64::NAN).is_err());
        assert_eq!(
            (neuron.v, neuron.theta, neuron.refractory_remaining),
            before
        );
    }
}
