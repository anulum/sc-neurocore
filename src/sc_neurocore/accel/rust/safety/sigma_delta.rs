// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! Fail-closed sampled asynchronous pulse sigma-delta transition.

const LIMIT: f64 = 1.0e12;

/// Complete sampled APSDM state and configuration.
#[derive(Debug, Clone)]
pub struct SigmaDeltaNeuron {
    pub sigma: f64,
    pub reconstruction: f64,
    pub delta: f64,
    pub tau_reconstruction: f64,
    pub dt: f64,
}
impl SigmaDeltaNeuron {
    /// Construct the documented numerical specialization.
    pub fn new() -> Self {
        Self {
            sigma: 0.0,
            reconstruction: 0.0,
            delta: 1.0,
            tau_reconstruction: 10.0,
            dt: 0.1,
        }
    }
    /// Advance atomically or return an error without mutation.
    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() || !validate_sigma_delta(self) {
            return Err("invalid SigmaDelta state or input");
        }
        let sigma = self.sigma + self.dt * current;
        let mut reconstruction = self.reconstruction * (-self.dt / self.tau_reconstruction).exp();
        let spike = sigma - reconstruction >= 0.5 * self.delta;
        if spike {
            reconstruction += self.delta;
        }
        if !sigma.is_finite()
            || !reconstruction.is_finite()
            || sigma.abs() > LIMIT
            || reconstruction.abs() > LIMIT
        {
            return Err("SigmaDelta candidate outside safety envelope");
        }
        self.sigma = sigma;
        self.reconstruction = reconstruction;
        Ok(i32::from(spike))
    }
    /// Clear dynamic state.
    pub fn reset(&mut self) {
        self.sigma = 0.0;
        self.reconstruction = 0.0;
    }
}
impl Default for SigmaDeltaNeuron {
    fn default() -> Self {
        Self::new()
    }
}
/// Validate complete source state and configuration.
pub fn validate_sigma_delta(s: &SigmaDeltaNeuron) -> bool {
    [
        s.sigma,
        s.reconstruction,
        s.delta,
        s.tau_reconstruction,
        s.dt,
    ]
    .iter()
    .all(|v| v.is_finite())
        && s.sigma.abs() <= LIMIT
        && s.reconstruction.abs() <= LIMIT
        && s.delta > 0.0
        && s.tau_reconstruction > 0.0
        && s.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_transition_is_atomic() {
        let mut s = SigmaDeltaNeuron {
            sigma: 0.49,
            ..SigmaDeltaNeuron::new()
        };
        assert_eq!(s.step(0.2), Ok(1));
        let before = (s.sigma, s.reconstruction);
        assert!(s.step(f64::NAN).is_err());
        assert_eq!((s.sigma, s.reconstruction), before);
    }
}
