// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for escape_rate

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct EscapeRateNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub rho_0: f64,
    pub delta_u: f64,
    pub resistance: f64,
    pub dt: f64,
}

impl EscapeRateNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0_f64,
            v_rest: -70.0_f64,
            v_reset: -70.0_f64,
            v_threshold: -50.0_f64,
            tau_m: 10.0_f64,
            rho_0: 0.001_f64,
            delta_u: 3.0_f64,
            resistance: 1.0_f64,
            dt: 1.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !i_ext.is_finite() {
            return Err("escape rate input current must be finite");
        }
        if !validate_escape_rate(self) {
            return Err("escape rate state parameters must be finite and positive");
        }

        let v_inf = self.v_rest + self.resistance * i_ext;
        let decay = (-self.dt / self.tau_m).exp();
        let next_v = v_inf + (self.v - v_inf) * decay;
        if !v_inf.is_finite() || !decay.is_finite() || !next_v.is_finite() {
            return Err("escape rate membrane candidate must remain finite");
        }
        let hazard = self.rho_0 * safe_exp((next_v - self.v_threshold) / self.delta_u) * self.dt;
        if !hazard.is_finite() || hazard < 0.0 {
            return Err("escape rate hazard must remain finite and non-negative");
        }
        let p_spike = -(-hazard).exp_m1();
        if !p_spike.is_finite() || !(0.0..=1.0).contains(&p_spike) {
            return Err("escape rate spike probability must remain finite and bounded");
        }
        if p_spike >= 1.0 {
            self.v = self.v_reset;
            return Ok(1);
        }
        self.v = next_v;
        Ok(0)
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
    }
}

pub fn validate_escape_rate(state: &EscapeRateNeuron) -> bool {
    state.v.is_finite()
        && state.v_rest.is_finite()
        && state.v_reset.is_finite()
        && state.v_threshold.is_finite()
        && state.tau_m.is_finite()
        && state.tau_m > 0.0
        && state.rho_0.is_finite()
        && state.rho_0 > 0.0
        && state.delta_u.is_finite()
        && state.delta_u > 0.0
        && state.resistance.is_finite()
        && state.resistance > 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
}

fn safe_exp(x: f64) -> f64 {
    x.clamp(-700.0, 700.0).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_rate_new() {
        let state = EscapeRateNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_escape_rate(&state));
    }

    #[test]
    fn test_escape_rate_step() {
        let mut state = EscapeRateNeuron::new();
        let spike = state.step(10.0).expect("valid step must succeed");
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_exact_flow_matches_closed_form() {
        let mut state = EscapeRateNeuron::new();
        state.v = -65.0;
        state.dt = 5.0;
        state.rho_0 = 1.0e-12;
        let current = 10.0;
        let v0 = state.v;
        let v_inf = state.v_rest + state.resistance * current;
        let euler =
            v0 + (-(v0 - state.v_rest) + state.resistance * current) / state.tau_m * state.dt;
        let expected = v_inf + (v0 - v_inf) * (-state.dt / state.tau_m).exp();

        let spike = state.step(current).expect("valid step must succeed");
        assert_eq!(spike, 0);
        assert!((state.v - expected).abs() < 1e-14);
        assert!((state.v - euler).abs() > 1e-3);
    }

    #[test]
    fn test_invalid_runtime_state_does_not_mutate_voltage() {
        let mut state = EscapeRateNeuron::new();
        state.v = -65.0;
        state.delta_u = 0.0;

        assert!(state.step(1.0).is_err());
        assert_eq!(state.v, -65.0);
    }

    #[test]
    fn test_non_finite_update_does_not_mutate_voltage() {
        let mut state = EscapeRateNeuron::new();
        state.v = -65.0;
        state.v_threshold = 1.0e308;
        state.resistance = 1.0e308;

        assert!(state.step(1.0e308).is_err());
        assert_eq!(state.v, -65.0);
    }

    #[test]
    fn test_non_finite_hazard_does_not_mutate_voltage() {
        let mut state = EscapeRateNeuron::new();
        state.v = -50.0;
        state.rho_0 = 1.0e308;
        state.dt = 1.0e308;

        assert!(state.step(0.0).is_err());
        assert_eq!(state.v, -50.0);
    }
}
