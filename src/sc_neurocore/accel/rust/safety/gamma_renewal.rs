// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for gamma_renewal

#![allow(non_snake_case)]

const DEFAULT_RNG_SEED: u64 = 0x4741_4d4d_415f_524e;

#[derive(Debug, Clone)]
pub struct GammaRenewalNeuron {
    pub rate_hz: f64,
    pub shape_k: usize,
    pub dt_ms: f64,
    pub time_since_spike_s: f64,
    pub rng_state: u64,
}

impl GammaRenewalNeuron {
    pub fn new() -> Self {
        Self::try_new(50.0, 3, 1.0, None).expect("default gamma-renewal parameters are valid")
    }

    pub fn try_new(
        rate_hz: f64,
        shape_k: usize,
        dt_ms: f64,
        rng_seed: Option<u64>,
    ) -> Result<Self, String> {
        validate_params(rate_hz, shape_k, dt_ms)?;
        Ok(Self {
            rate_hz,
            shape_k,
            dt_ms,
            time_since_spike_s: 0.0,
            rng_state: rng_seed.unwrap_or(DEFAULT_RNG_SEED),
        })
    }

    pub fn step(&mut self, rate_override: f64) -> Result<i32, String> {
        if !rate_override.is_finite() {
            return Err("rate_override must be finite".to_string());
        }

        let rate_hz = if rate_override < 0.0 {
            self.rate_hz
        } else {
            rate_override
        };
        if rate_hz < 0.0 {
            return Err("effective rate_hz must be non-negative".to_string());
        }

        self.time_since_spike_s += self.dt_ms / 1000.0;
        let p_spike = self.spike_probability_at(self.time_since_spike_s, rate_hz)?;
        if self.next_unit_interval() < p_spike {
            self.time_since_spike_s = 0.0;
            Ok(1)
        } else {
            Ok(0)
        }
    }

    pub fn spike_probability_at(&self, elapsed_s: f64, rate_hz: f64) -> Result<f64, String> {
        if !elapsed_s.is_finite() || elapsed_s < 0.0 {
            return Err("elapsed_s must be finite and non-negative".to_string());
        }
        if !rate_hz.is_finite() || rate_hz < 0.0 {
            return Err("rate_hz must be finite and non-negative".to_string());
        }
        if elapsed_s < 1.0e-12 || rate_hz == 0.0 {
            return Ok(0.0);
        }

        let k = self.shape_k;
        let lambda = k as f64 * rate_hz;
        let x = lambda * elapsed_s;
        let log_f =
            k as f64 * lambda.ln() + (k as f64 - 1.0) * elapsed_s.ln() - x - log_gamma_int(k);
        let density = log_f.clamp(-50.0, 50.0).exp();
        let survival = gamma_survival(k, x)?.max(1.0e-15);
        let hazard = density / survival;
        Ok(-(-(hazard * self.dt_ms / 1000.0)).exp_m1())
    }

    pub fn reset(&mut self) {
        self.time_since_spike_s = 0.0;
    }

    fn next_unit_interval(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.rng_state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

fn validate_params(rate_hz: f64, shape_k: usize, dt_ms: f64) -> Result<(), String> {
    if !rate_hz.is_finite() || rate_hz <= 0.0 {
        return Err("rate_hz must be finite and positive".to_string());
    }
    if shape_k == 0 {
        return Err("shape_k must be a positive integer".to_string());
    }
    if !dt_ms.is_finite() || dt_ms <= 0.0 {
        return Err("dt_ms must be finite and positive".to_string());
    }
    Ok(())
}

pub fn log_gamma_int(k: usize) -> f64 {
    if k <= 1 {
        0.0
    } else {
        (1..k).map(|i| (i as f64).ln()).sum()
    }
}

pub fn gamma_survival(k: usize, x: f64) -> Result<f64, String> {
    if k == 0 {
        return Err("shape_k must be a positive integer".to_string());
    }
    if !x.is_finite() {
        return Err("x must be finite".to_string());
    }
    if x < 0.0 {
        return Ok(1.0);
    }

    let mut series = 1.0;
    let mut term = 1.0;
    for i in 1..k {
        term *= x / i as f64;
        series += term;
    }
    Ok((-x).exp() * series)
}

pub fn validate_gamma_renewal(state: &GammaRenewalNeuron) -> bool {
    validate_params(state.rate_hz, state.shape_k, state.dt_ms).is_ok()
        && state.time_since_spike_s.is_finite()
        && state.time_since_spike_s >= 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_renewal_new() {
        let state = GammaRenewalNeuron::new();
        assert!(validate_gamma_renewal(&state));
        assert_eq!(state.rate_hz, 50.0);
        assert_eq!(state.shape_k, 3);
        assert_eq!(state.dt_ms, 1.0);
    }

    #[test]
    fn test_gamma_renewal_step() {
        let mut state = GammaRenewalNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_gamma_renewal_validates_parameters_and_rate_override() {
        assert!(GammaRenewalNeuron::try_new(0.0, 3, 1.0, None).is_err());
        assert!(GammaRenewalNeuron::try_new(50.0, 0, 1.0, None).is_err());
        assert!(GammaRenewalNeuron::try_new(50.0, 3, 0.0, None).is_err());

        let mut state = GammaRenewalNeuron::new();
        assert!(state.step(f64::NAN).is_err());
    }

    #[test]
    fn test_gamma_survival_matches_integer_shape_series() {
        let x = 0.15_f64;
        let expected = (-x).exp() * (1.0 + x + x * x / 2.0);
        assert!((gamma_survival(3, x).unwrap() - expected).abs() < 1.0e-15);
    }

    #[test]
    fn test_gamma_renewal_high_rate_forces_spike_and_resets_elapsed_time() {
        let mut state = GammaRenewalNeuron::try_new(50.0, 1, 1.0, Some(7)).unwrap();
        let spike = state.step(2_000.0).unwrap();

        assert_eq!(spike, 1);
        assert_eq!(state.time_since_spike_s, 0.0);
    }

    #[test]
    fn test_gamma_renewal_zero_rate_override_never_spikes_but_advances_time() {
        let mut state = GammaRenewalNeuron::try_new(50.0, 3, 1.0, Some(7)).unwrap();
        let spike = state.step(0.0).unwrap();

        assert_eq!(spike, 0);
        assert!((state.time_since_spike_s - 0.001).abs() < 1.0e-15);
    }
}
