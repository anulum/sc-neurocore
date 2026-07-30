// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! Fail-closed Rust safety implementation of source MAT(1).

const V_MIN: f64 = -200.0;
const V_MAX: f64 = 200.0;
const THETA_MAX: f64 = 1.0e9;

/// Complete MAT(1) state and configuration.
#[derive(Debug, Clone)]
pub struct NonResettingLIFNeuron {
    /// Non-resetting membrane voltage relative to rest, in mV.
    pub v: f64,
    /// Single spike-history threshold contribution, in mV.
    pub theta: f64,
    /// Absolute refractory time remaining, in ms.
    pub refractory_remaining: f64,
    /// Baseline threshold, in mV.
    pub omega: f64,
    /// Membrane time constant, in ms.
    pub tau_m: f64,
    /// MAT(1) threshold time constant, in ms.
    pub tau_theta: f64,
    /// Threshold increment per event, in mV.
    pub alpha: f64,
    /// Membrane resistance, in MOhm.
    pub resistance: f64,
    /// Absolute refractory interval, in ms.
    pub refractory_period: f64,
    /// Sample interval, in ms.
    pub dt: f64,
}

impl NonResettingLIFNeuron {
    /// Construct the enrolled MAT(1) specialization.
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

    /// Advance one atomic source-model sample.
    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() || !validate_non_resetting_lif(self) {
            return Err("invalid MAT(1) state, configuration, or current");
        }
        let v = self.v + self.dt * (-self.v + self.resistance * current) / self.tau_m;
        let mut theta = self.theta * (-self.dt / self.tau_theta).exp();
        let mut refractory = (self.refractory_remaining - self.dt).max(0.0);
        if !v.is_finite()
            || !theta.is_finite()
            || !refractory.is_finite()
            || !(V_MIN..=V_MAX).contains(&v)
            || !(0.0..=THETA_MAX).contains(&theta)
        {
            return Err("MAT(1) candidate outside safety envelope");
        }
        let spike = refractory == 0.0 && v >= self.omega + theta;
        if spike {
            theta += self.alpha;
            refractory = self.refractory_period;
        }
        if !theta.is_finite() || theta > THETA_MAX {
            return Err("MAT(1) post-spike threshold outside safety envelope");
        }
        self.v = v;
        self.theta = theta;
        self.refractory_remaining = refractory;
        Ok(i32::from(spike))
    }

    /// Restore zero-rest source state.
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

/// Return whether all source state and configuration invariants hold.
pub fn validate_non_resetting_lif(s: &NonResettingLIFNeuron) -> bool {
    [
        s.v,
        s.theta,
        s.refractory_remaining,
        s.omega,
        s.tau_m,
        s.tau_theta,
        s.alpha,
        s.resistance,
        s.refractory_period,
        s.dt,
    ]
    .iter()
    .all(|v| v.is_finite())
        && (V_MIN..=V_MAX).contains(&s.v)
        && (0.0..=THETA_MAX).contains(&s.theta)
        && (-THETA_MAX..=THETA_MAX).contains(&s.omega)
        && (0.0..=THETA_MAX).contains(&s.alpha)
        && s.tau_m > 0.0
        && s.tau_theta > 0.0
        && s.resistance > 0.0
        && s.refractory_period >= 0.0
        && s.dt > 0.0
        && (0.0..=s.refractory_period).contains(&s.refractory_remaining)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_event_is_non_resetting() {
        let mut n = NonResettingLIFNeuron {
            v: 20.0,
            ..NonResettingLIFNeuron::new()
        };
        assert_eq!(n.step(0.0), Ok(1));
        assert!(n.v > 19.0);
        assert_eq!(n.theta, 37.0);
        assert_eq!(n.refractory_remaining, 2.0);
    }
    #[test]
    fn invalid_input_is_atomic() {
        let mut n = NonResettingLIFNeuron::new();
        let before = (n.v, n.theta, n.refractory_remaining);
        assert!(n.step(f64::NAN).is_err());
        assert_eq!((n.v, n.theta, n.refractory_remaining), before);
    }
}
