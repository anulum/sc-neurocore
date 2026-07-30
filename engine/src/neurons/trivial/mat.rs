// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Kobayashi 2009 non-resetting MAT* neuron

//! Source-faithful non-resetting MAT* adaptive-threshold neuron.

const V_MIN: f64 = -200.0;
const V_MAX: f64 = 200.0;
const THETA_MAX: f64 = 1.0e9;

/// Complete MAT* dynamic state and configuration.
///
/// Voltage is relative to rest and is never reset. Units are millivolts,
/// milliseconds, nanoamps, and megaohms. The defaults are the regular-spiking
/// example in Kobayashi, Tsubo, and Shinomoto (2009), not a universal cell fit.
#[derive(Clone, Debug, PartialEq)]
pub struct MATNeuron {
    /// Membrane voltage relative to rest, in mV.
    pub v: f64,
    /// Fast spike-history threshold contribution, in mV.
    pub theta1: f64,
    /// Slow spike-history threshold contribution, in mV.
    pub theta2: f64,
    /// Remaining absolute refractory interval, in ms.
    pub refractory_remaining: f64,
    /// Baseline threshold omega, in mV.
    pub omega: f64,
    /// Membrane time constant, in ms.
    pub tau_m: f64,
    /// Fast threshold-history time constant, in ms.
    pub tau_1: f64,
    /// Slow threshold-history time constant, in ms.
    pub tau_2: f64,
    /// Fast post-spike threshold increment, in mV.
    pub alpha_1: f64,
    /// Slow post-spike threshold increment, in mV.
    pub alpha_2: f64,
    /// Membrane resistance, in megaohms.
    pub resistance: f64,
    /// Absolute refractory duration, in ms.
    pub refractory_period: f64,
    /// Euler integration step, in ms.
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

    /// Return the paper's intrinsically-bursting example profile.
    #[must_use]
    pub fn intrinsically_bursting() -> Self {
        Self {
            omega: 26.0,
            alpha_1: 1.7,
            alpha_2: 2.0,
            ..Self::new()
        }
    }

    /// Return the paper's fast-spiking example profile.
    #[must_use]
    pub fn fast_spiking() -> Self {
        Self {
            omega: 11.0,
            alpha_1: 10.0,
            alpha_2: 0.002,
            ..Self::new()
        }
    }

    /// Return the instantaneous adaptive threshold, in mV.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.omega + self.theta1 + self.theta2
    }

    /// Return whether all state and configuration invariants hold.
    #[must_use]
    pub fn validate(&self) -> bool {
        let finite = [
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
        .all(|value| value.is_finite());
        finite
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

    /// Advance one atomic MAT* step.
    ///
    /// Voltage uses forward Euler and both threshold-history terms use exact
    /// exponential decay. A spike never resets voltage. Invalid input or a
    /// candidate outside the safety envelope returns an error without mutation.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() || !self.validate() {
            return Err("invalid MAT state, configuration, or current");
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
            return Err("MAT candidate outside safety envelope");
        }
        let spike = refractory == 0.0 && v >= self.omega + theta1 + theta2;
        if spike {
            theta1 += self.alpha_1;
            theta2 += self.alpha_2;
            refractory = self.refractory_period;
            if theta1 > THETA_MAX || theta2 > THETA_MAX {
                return Err("MAT post-spike threshold outside safety envelope");
            }
        }
        self.v = v;
        self.theta1 = theta1;
        self.theta2 = theta2;
        self.refractory_remaining = refractory;
        Ok(i32::from(spike))
    }

    /// Advance one engine-adapter step; invalid state returns no event.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Reset dynamic state while preserving the configured profile.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_step_is_non_resetting() {
        let mut neuron = MATNeuron {
            v: 20.0,
            ..MATNeuron::new()
        };
        let expected = 20.0 + neuron.dt * (-20.0) / neuron.tau_m;
        assert_eq!(neuron.try_step(0.0), Ok(1));
        assert_eq!(neuron.v, expected);
        assert_eq!(neuron.theta1, 37.0);
        assert_eq!(neuron.theta2, 2.0);
    }

    #[test]
    fn source_reemits_after_refractory() {
        let mut neuron = MATNeuron {
            v: 25.0,
            omega: 1.0,
            alpha_1: 0.0,
            alpha_2: 0.0,
            tau_m: 1.0e9,
            refractory_period: 2.0,
            dt: 0.5,
            ..MATNeuron::new()
        };
        assert_eq!(neuron.try_step(0.0), Ok(1));
        assert_eq!(
            (0..4)
                .map(|_| neuron.try_step(0.0).unwrap())
                .collect::<Vec<_>>(),
            vec![0, 0, 0, 1]
        );
    }

    #[test]
    fn failure_is_atomic() {
        let mut neuron = MATNeuron::new();
        let before = neuron.clone();
        assert!(neuron.try_step(f64::NAN).is_err());
        assert_eq!(neuron, before);
    }
}
