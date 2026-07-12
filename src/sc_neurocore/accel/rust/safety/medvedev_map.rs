// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety kernel for the Medvedev 2005 first-return map

/// Calibrated slow-calcium first-return map derived from Medvedev (2005).
#[derive(Debug, Clone)]
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

    fn u_0(&self) -> f64 {
        self.beta_0 / (self.delta - self.beta_0)
    }

    fn u_hc(&self) -> f64 {
        self.beta_hc / (self.delta - self.beta_hc)
    }

    fn u_sn(&self) -> f64 {
        self.beta_sn / (self.delta - self.beta_sn)
    }

    fn candidate(&self, current: f64) -> Option<f64> {
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
                    return None;
                }
                let scale = (self.homoclinic_exponent * log_argument.ln()).exp();
                scale * (u_1 - self.f_1) + self.f_1
            };
            inner_return + self.input_gain * current
        } else {
            self.u_sn()
        };
        candidate.is_finite().then_some(candidate)
    }

    /// Advance one checked return. Rejected input leaves `u` unchanged.
    pub fn try_step(&mut self, current: f64) -> Option<i32> {
        if !validate_medvedev_map(self) || !current.is_finite() {
            return None;
        }
        let event = i32::from(self.u <= self.u_hc());
        let candidate = self.candidate(current)?;
        self.u = candidate;
        Some(event)
    }

    /// Compatibility surface: invalid input emits no event and preserves state.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    pub fn reset(&mut self) {
        if validate_parameters(self) {
            self.u = self.u_sn();
        }
    }
}

impl Default for MedvedevMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_parameters(state: &MedvedevMapNeuron) -> bool {
    state.beta_0.is_finite()
        && state.beta_hc.is_finite()
        && state.beta_sn.is_finite()
        && state.delta.is_finite()
        && state.decay_t0.is_finite()
        && state.alpha_t0.is_finite()
        && state.f_0.is_finite()
        && state.f_1.is_finite()
        && state.homoclinic_exponent.is_finite()
        && state.d.is_finite()
        && state.input_gain.is_finite()
        && 0.0 < state.beta_0
        && state.beta_0 < state.beta_sn
        && state.beta_sn < state.beta_hc
        && state.beta_hc < state.delta
        && 0.0 < state.decay_t0
        && state.decay_t0 < 1.0
        && 0.0 < state.alpha_t0
        && state.alpha_t0 < 1.0
        && 0.0 <= state.f_1
        && state.f_1 < state.f_0
        && state.homoclinic_exponent > 0.0
        && state.d > 0.0
        && state.input_gain >= 0.0
}

pub fn validate_medvedev_map(state: &MedvedevMapNeuron) -> bool {
    state.u.is_finite() && validate_parameters(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibrated_goldens_match() {
        for (current, expected_events, expected_final) in [
            (0.0, 100, 0.194_484_917_610_024_04),
            (2.0, 75, 0.251_407_883_672_443_6),
        ] {
            let mut state = MedvedevMapNeuron::new();
            let events: i32 = (0..100)
                .map(|_| state.try_step(current).expect("finite source regime"))
                .sum();
            assert_eq!(events, expected_events, "current={current}");
            assert!((state.u - expected_final).abs() < 1.0e-14);
        }
    }

    #[test]
    fn invalid_current_preserves_state() {
        let mut state = MedvedevMapNeuron::new();
        let before = state.u;
        assert_eq!(state.try_step(f64::NAN), None);
        assert_eq!(state.u, before);
    }

    #[test]
    fn reset_preserves_parameters() {
        let mut state = MedvedevMapNeuron::new();
        state.u = 0.2;
        state.input_gain = 0.02;
        state.reset();
        assert_eq!(state.u, state.u_sn());
        assert_eq!(state.input_gain, 0.02);
    }
}
