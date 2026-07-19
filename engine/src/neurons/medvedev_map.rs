// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Medvedev first-return map neuron

//! Medvedev first-return map neuron.

/// Medvedev (2005) calibrated slow-calcium first-return map.
#[derive(Clone, Debug)]
pub struct MedvedevMapNeuron {
    pub u: f64,
    pub beta_0: f64,
    pub beta_hc: f64,
    pub beta_sn: f64,
    pub delta: f64,
    pub decay_t0: f64,
    pub alpha_t0: f64,
    pub f_0: f64,
    pub f_1: f64,
    pub homoclinic_exponent: f64,
    pub d: f64,
    pub input_gain: f64,
}

impl MedvedevMapNeuron {
    pub fn new() -> Self {
        Self {
            u: 0.251_407_883_672_443_6,
            beta_0: 0.0015,
            beta_hc: 0.00203,
            beta_sn: 0.002_009_000_318_382_601,
            delta: 0.01,
            decay_t0: 0.990_356_335_578_673_4,
            alpha_t0: 0.009_690_465_686_585_3,
            f_0: 1.471_354_142_980_228_6,
            f_1: 0.182_015_278_714_566_5,
            homoclinic_exponent: 0.021_492_989_913_392_21,
            d: 2_271.192_797_740_406,
            input_gain: 0.01,
        }
    }

    fn parameters_are_valid(&self) -> bool {
        self.beta_0.is_finite()
            && self.beta_hc.is_finite()
            && self.beta_sn.is_finite()
            && self.delta.is_finite()
            && self.decay_t0.is_finite()
            && self.alpha_t0.is_finite()
            && self.f_0.is_finite()
            && self.f_1.is_finite()
            && self.homoclinic_exponent.is_finite()
            && self.d.is_finite()
            && self.input_gain.is_finite()
            && 0.0 < self.beta_0
            && self.beta_0 < self.beta_sn
            && self.beta_sn < self.beta_hc
            && self.beta_hc < self.delta
            && 0.0 < self.decay_t0
            && self.decay_t0 < 1.0
            && 0.0 < self.alpha_t0
            && self.alpha_t0 < 1.0
            && 0.0 <= self.f_1
            && self.f_1 < self.f_0
            && self.homoclinic_exponent > 0.0
            && self.d > 0.0
            && self.input_gain >= 0.0
    }

    fn u_0(&self) -> f64 {
        self.beta_0 / (self.delta - self.beta_0)
    }

    fn u_hc(&self) -> f64 {
        self.beta_hc / (self.delta - self.beta_hc)
    }

    fn u_sn(&self) -> f64 {
        self.beta_sn / (self.delta - self.beta_sn)
    }

    fn candidate(&self, current: f64) -> Result<f64, &'static str> {
        let candidate = if self.u <= self.u_0() {
            self.decay_t0 * self.u + (1.0 - self.decay_t0) * self.f_0 + self.input_gain * current
        } else if self.u <= self.u_hc() {
            let u_1 = (1.0 - self.alpha_t0) * self.u + self.alpha_t0 * self.f_0;
            let gap = self.beta_hc - self.delta * u_1 / (1.0 + u_1);
            let inner_return = if gap <= 0.0 {
                self.f_1
            } else {
                let log_argument = self.d * gap;
                if !log_argument.is_finite() || log_argument <= 0.0 {
                    return Err("invalid Medvedev homoclinic log argument");
                }
                let scale = (self.homoclinic_exponent * log_argument.ln()).exp();
                scale * (u_1 - self.f_1) + self.f_1
            };
            inner_return + self.input_gain * current
        } else {
            self.u_sn()
        };
        if !candidate.is_finite() {
            return Err("invalid Medvedev first-return candidate");
        }
        Ok(candidate)
    }

    /// Checked source-derived update; a rejected step leaves the state intact.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !self.u.is_finite() || !self.parameters_are_valid() {
            return Err("invalid Medvedev first-return runtime state");
        }
        if !current.is_finite() {
            return Err("invalid Medvedev first-return current");
        }
        let event = i32::from(self.u <= self.u_hc());
        let candidate = self.candidate(current)?;
        self.u = candidate;
        Ok(event)
    }

    /// Legacy infallible engine-class update. Invalid input leaves the state
    /// unchanged and emits no event; the checked batch API reports the error.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Run checked first-return iterations, returning the `u` trace and events.
    pub fn simulate(
        &mut self,
        n_steps: usize,
        current: f64,
    ) -> Result<(Vec<f64>, i64), &'static str> {
        let mut trace = Vec::with_capacity(n_steps);
        let mut events = 0_i64;
        for _ in 0..n_steps {
            events += i64::from(self.try_step(current)?);
            trace.push(self.u);
        }
        Ok((trace, events))
    }

    pub fn reset(&mut self) {
        if self.parameters_are_valid() {
            self.u = self.u_sn();
        }
    }
}
impl Default for MedvedevMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn medvedev_matches_calibrated_first_return_goldens() {
        let mut zero_current = MedvedevMapNeuron::new();
        let (trace, events) = zero_current
            .simulate(100, 0.0)
            .expect("calibrated source regime must remain finite");
        let mean = trace.iter().sum::<f64>() / trace.len() as f64;
        assert_eq!(events, 100);
        assert!((zero_current.u - 0.194_484_917_610_024_04).abs() < 1.0e-14);
        assert!((mean - 0.216_230_983_622_399_98).abs() < 1.0e-14);

        let mut driven = MedvedevMapNeuron::new();
        let (_trace, events) = driven
            .simulate(100, 2.0)
            .expect("calibrated driven regime must remain finite");
        assert_eq!(events, 75);
        assert_eq!(driven.u, driven.u_sn());
    }

    #[test]
    fn medvedev_rejects_invalid_runtime_without_mutation() {
        let mut neuron = MedvedevMapNeuron::new();
        let initial = neuron.u;
        assert!(neuron.try_step(f64::NAN).is_err());
        assert_eq!(neuron.u, initial);

        neuron.d = f64::INFINITY;
        assert!(neuron.try_step(0.0).is_err());
        assert_eq!(neuron.u, initial);
    }

    #[test]
    fn medvedev_reset_preserves_calibration() {
        let mut neuron = MedvedevMapNeuron::new();
        neuron.u = 0.2;
        neuron.input_gain = 0.02;
        neuron.reset();
        assert_eq!(neuron.u, neuron.u_sn());
        assert_eq!(neuron.input_gain, 0.02);
    }
}
