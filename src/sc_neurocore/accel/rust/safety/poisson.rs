// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for poisson

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct PoissonNeuron {
    pub rate_hz: f64,
    pub dt_ms: f64,
    pub _rng: f64,
}

impl PoissonNeuron {
    pub fn new() -> Self {
        Self {
            rate_hz: 100.0_f64,
            dt_ms: 1.0_f64,
            _rng: 0.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !i_ext.is_finite() {
            return Err("poisson rate override must be finite");
        }
        if !validate_poisson(self) {
            return Err("poisson rate and timestep must be finite with non-negative rate and positive timestep");
        }

        let rate_hz = if i_ext < 0.0 { self.rate_hz } else { i_ext };
        if !rate_hz.is_finite() || rate_hz < 0.0 {
            return Err("poisson active rate must be finite and non-negative");
        }
        let hazard = rate_hz * self.dt_ms / 1000.0;
        if !hazard.is_finite() || hazard < 0.0 {
            return Err("poisson interval hazard must remain finite and non-negative");
        }
        let p_spike = -(-hazard).exp_m1();
        if !p_spike.is_finite() || !(0.0..=1.0).contains(&p_spike) {
            return Err("poisson spike probability must remain finite and bounded");
        }
        if p_spike >= 1.0 {
            return Ok(1);
        }
        Ok(0)
    }

    pub fn reset(&mut self) {
        // Stateless model: reset is intentionally a no-op.
    }
}

pub fn validate_poisson(state: &PoissonNeuron) -> bool {
    state.rate_hz.is_finite()
        && state.rate_hz >= 0.0
        && state.dt_ms.is_finite()
        && state.dt_ms > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_new() {
        let state = PoissonNeuron::new();
        assert!(validate_poisson(&state));
    }

    #[test]
    fn test_poisson_step() {
        let mut state = PoissonNeuron::new();
        let spike = state.step(10.0).expect("valid step must succeed");
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_high_rate_saturates() {
        let mut state = PoissonNeuron::new();
        assert_eq!(state.step(1.0e9), Ok(1));
    }

    #[test]
    fn test_invalid_runtime_state_fails() {
        let mut state = PoissonNeuron::new();
        state.dt_ms = 0.0;
        assert!(state.step(-1.0).is_err());
    }

    #[test]
    fn test_non_finite_rate_override_fails() {
        let mut state = PoissonNeuron::new();
        assert!(state.step(f64::INFINITY).is_err());
    }

    #[test]
    fn test_non_finite_interval_hazard_fails() {
        let mut state = PoissonNeuron::new();
        state.rate_hz = 1.0e308;
        state.dt_ms = 1.0e308;
        assert!(state.step(-1.0).is_err());
    }
}
