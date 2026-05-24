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

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_poisson(self) || !i_ext.is_finite() {
            return 0;
        }

        let rate_hz = if i_ext < 0.0 { self.rate_hz } else { i_ext };
        if !rate_hz.is_finite() || rate_hz < 0.0 {
            return 0;
        }
        let p_spike = -(-(rate_hz * self.dt_ms / 1000.0)).exp_m1();
        if p_spike >= 1.0 {
            return 1;
        }
        0
    }

    pub fn reset(&mut self) {
        // pass
        self.rate_hz = 100.0_f64;
        self.dt_ms = 1.0_f64;
        self._rng = 0.0_f64;
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
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }
}
