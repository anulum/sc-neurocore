// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sigmoid rate neuron model

/// Sigmoid rate neuron — Wilson-Cowan-style single unit.
#[derive(Clone, Debug)]
pub struct SigmoidRateNeuron {
    pub r: f64,
    pub tau: f64,
    pub beta: f64,
    pub theta: f64,
    pub dt: f64,
}

impl SigmoidRateNeuron {
    /// Construct the maintained factory-default rate unit.
    pub fn new() -> Self {
        Self::with_parameters(0.0, 10.0, 1.0, 0.0, 0.1)
            .expect("the factory-default sigmoid-rate contract is valid")
    }

    /// Construct a fully configurable, validated sigmoid-rate unit.
    pub fn with_parameters(
        r: f64,
        tau: f64,
        beta: f64,
        theta: f64,
        dt: f64,
    ) -> Result<Self, String> {
        let neuron = Self {
            r,
            tau,
            beta,
            theta,
            dt,
        };
        neuron.validate()?;
        Ok(neuron)
    }

    /// Validate the complete mutable numeric contract.
    pub fn validate(&self) -> Result<(), String> {
        if !self.r.is_finite()
            || !(0.0..=1.0).contains(&self.r)
            || !self.tau.is_finite()
            || self.tau <= 0.0
            || !self.beta.is_finite()
            || !self.theta.is_finite()
            || !self.dt.is_finite()
            || self.dt <= 0.0
        {
            return Err(
                "sigmoid-rate state and parameters must be finite, with r in [0,1] and positive tau/dt"
                    .into(),
            );
        }
        Ok(())
    }

    /// Advance one step, preserving the previous state when validation fails.
    pub fn try_step(&mut self, current: f64) -> Result<f64, String> {
        self.validate()?;
        if !current.is_finite() {
            return Err("sigmoid-rate current must be finite".into());
        }
        let target = stable_sigmoid(self.beta, current, self.theta)?;
        let decay = (-self.dt / self.tau).exp();
        let candidate = decay * self.r + (1.0 - decay) * target;
        if !candidate.is_finite() || !(0.0..=1.0).contains(&candidate) {
            return Err("sigmoid-rate exact relaxation left the finite unit interval".into());
        }
        self.r = candidate;
        Ok(candidate)
    }

    /// Advance one step through the legacy non-throwing engine boundary.
    pub fn step(&mut self, current: f64) -> f64 {
        self.try_step(current).unwrap_or(self.r)
    }

    /// Restore the dynamic rate state without changing configured parameters.
    pub fn reset(&mut self) {
        self.r = 0.0;
    }
}

fn stable_sigmoid(beta: f64, current: f64, theta: f64) -> Result<f64, String> {
    let argument = beta * (current - theta);
    if argument.is_infinite() {
        return Ok(if argument.is_sign_positive() {
            1.0
        } else {
            0.0
        });
    }
    if !argument.is_finite() {
        return Err("sigmoid-rate transfer argument must be finite or saturating".into());
    }
    if argument >= 0.0 {
        Ok(1.0 / (1.0 + (-argument).exp()))
    } else {
        let exponential = argument.exp();
        Ok(exponential / (1.0 + exponential))
    }
}
impl Default for SigmoidRateNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_rate() {
        let mut n = SigmoidRateNeuron::new();
        for _ in 0..100 {
            n.step(5.0);
        }
        assert!(n.r > 0.5);
    }

    #[test]
    fn sigmoid_rate_matches_python_exact_relaxation_golden() {
        let mut neuron = SigmoidRateNeuron::with_parameters(0.25, 10.0, 2.0, 1.0, 0.5).unwrap();
        let expected = [
            0.2857007338135623,
            0.3196603222932904,
            0.3519636820991432,
            0.38269158845670403,
            0.41192087713731845,
            0.43972463658754457,
        ];
        for target in expected {
            let rate = neuron.try_step(3.0).unwrap();
            assert!((rate - target).abs() <= 2.0e-15, "{rate} != {target}");
        }
    }

    #[test]
    fn sigmoid_rate_exact_relaxation_is_bounded_for_large_timestep() {
        let mut neuron = SigmoidRateNeuron::with_parameters(1.0, 0.1, 1.0, 0.0, 5.0).unwrap();
        let rate = neuron.try_step(-100.0).unwrap();
        assert!((rate - 1.9287498479639178e-22).abs() <= 1.0e-36);
        assert!((0.0..=1.0).contains(&rate));
    }

    #[test]
    fn sigmoid_rate_rejects_invalid_contract_without_mutation() {
        let invalid_contracts = [
            (-0.1, 10.0, 1.0, 0.0, 0.1),
            (1.1, 10.0, 1.0, 0.0, 0.1),
            (0.0, 0.0, 1.0, 0.0, 0.1),
            (0.0, 10.0, f64::NAN, 0.0, 0.1),
            (0.0, 10.0, 1.0, f64::INFINITY, 0.1),
            (0.0, 10.0, 1.0, 0.0, -0.1),
        ];
        for (r, tau, beta, theta, dt) in invalid_contracts {
            assert!(SigmoidRateNeuron::with_parameters(r, tau, beta, theta, dt).is_err());
        }

        let mut neuron = SigmoidRateNeuron::with_parameters(0.25, 10.0, 2.0, 1.0, 0.5).unwrap();
        let before = neuron.r;
        assert!(neuron.try_step(f64::NAN).is_err());
        assert_eq!(neuron.r, before);
        neuron.tau = 0.0;
        assert!(neuron.try_step(3.0).is_err());
        assert_eq!(neuron.r, before);
    }

    #[test]
    fn sigmoid_rate_saturates_extreme_finite_drive() {
        let mut high = SigmoidRateNeuron::with_parameters(0.0, 10.0, 1.0e308, 0.0, 0.1).unwrap();
        let mut low = high.clone();
        assert!(high.try_step(1.0e308).unwrap() > 0.0);
        assert_eq!(low.try_step(-1.0e308).unwrap(), 0.0);
    }

    #[test]
    fn sigmoid_rate_reset_preserves_configuration() {
        let mut neuron = SigmoidRateNeuron::with_parameters(0.25, 7.0, 2.5, -0.4, 0.2).unwrap();
        neuron.try_step(3.0).unwrap();
        neuron.reset();
        assert_eq!(neuron.r, 0.0);
        assert_eq!(
            (neuron.tau, neuron.beta, neuron.theta, neuron.dt),
            (7.0, 2.5, -0.4, 0.2)
        );
    }

    #[test]
    fn sigmoid_rate_legacy_step_fails_closed() {
        let mut neuron = SigmoidRateNeuron::with_parameters(0.25, 10.0, 2.0, 1.0, 0.5).unwrap();
        assert_eq!(neuron.step(f64::NAN), 0.25);
        assert_eq!(neuron.r, 0.25);
    }
}
