// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! Sampled Yoon asynchronous pulse sigma-delta encoder.

const STATE_LIMIT: f64 = 1.0e12;

/// Integrating prefilter and local reconstruction-feedback state.
#[derive(Clone, Debug)]
pub struct SigmaDeltaNeuron {
    /// Integrating prefilter output.
    pub sigma: f64,
    /// Locally reconstructed signal.
    pub reconstruction: f64,
    /// Reconstruction quantum; upper threshold is half this value.
    pub delta: f64,
    /// Exponential reconstruction time constant.
    pub tau_reconstruction: f64,
    /// Discrete sample interval.
    pub dt: f64,
}

impl SigmaDeltaNeuron {
    /// Construct the documented source-equation specialization.
    pub fn new() -> Self {
        Self {
            sigma: 0.0,
            reconstruction: 0.0,
            delta: 1.0,
            tau_reconstruction: 10.0,
            dt: 0.1,
        }
    }

    /// Validate complete state and configuration.
    pub fn validate(&self) -> bool {
        [
            self.sigma,
            self.reconstruction,
            self.delta,
            self.tau_reconstruction,
            self.dt,
        ]
        .iter()
        .all(|value| value.is_finite())
            && self.sigma.abs() <= STATE_LIMIT
            && self.reconstruction.abs() <= STATE_LIMIT
            && self.delta > 0.0
            && self.tau_reconstruction > 0.0
            && self.dt > 0.0
    }

    /// Advance one atomic sampled APSDM transition.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() || !self.validate() {
            return Err("invalid SigmaDelta state, configuration, or current");
        }
        let sigma = self.sigma + self.dt * current;
        let mut reconstruction = self.reconstruction * (-self.dt / self.tau_reconstruction).exp();
        let spike = sigma - reconstruction >= 0.5 * self.delta;
        if spike {
            reconstruction += self.delta;
        }
        if !sigma.is_finite()
            || !reconstruction.is_finite()
            || sigma.abs() > STATE_LIMIT
            || reconstruction.abs() > STATE_LIMIT
        {
            return Err("SigmaDelta candidate outside safety envelope");
        }
        self.sigma = sigma;
        self.reconstruction = reconstruction;
        Ok(i32::from(spike))
    }

    /// Engine adapter: invalid state emits no event.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Clear dynamic state while retaining configuration.
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_step_and_atomic_failure() {
        let mut neuron = SigmaDeltaNeuron {
            sigma: 0.49,
            ..SigmaDeltaNeuron::new()
        };
        assert_eq!(neuron.try_step(0.2), Ok(1));
        assert_eq!(neuron.sigma, 0.51);
        let before = (neuron.sigma, neuron.reconstruction);
        assert!(neuron.try_step(f64::NAN).is_err());
        assert_eq!((neuron.sigma, neuron.reconstruction), before);
    }
}
