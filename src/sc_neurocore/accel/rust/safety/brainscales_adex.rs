// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for brainscales_adex

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct BrainScaleSAdExNeuron {
    pub v: f64,
    pub w: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub delta_t: f64,
    pub v_rh: f64,
    pub tau: f64,
    pub tau_w: f64,
    pub a: f64,
    pub b: f64,
    pub hw_speedup: f64,
    pub dt: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrainScaleSAdExError {
    InvalidInput,
    InvalidState,
    NonFiniteUpdate,
}

impl BrainScaleSAdExNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0_f64,
            w: 0.0_f64,
            v_rest: -65.0_f64,
            v_reset: -68.0_f64,
            v_threshold: -50.0_f64,
            delta_t: 2.0_f64,
            v_rh: -55.0_f64,
            tau: 20.0_f64,
            tau_w: 100.0_f64,
            a: 0.5_f64,
            b: 7.0_f64,
            hw_speedup: 1000.0_f64,
            dt: 0.1_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, BrainScaleSAdExError> {
        if !i_ext.is_finite() {
            return Err(BrainScaleSAdExError::InvalidInput);
        }
        if !validate_brainscales_adex(self) {
            return Err(BrainScaleSAdExError::InvalidState);
        }

        let dt_hw = self.dt * self.hw_speedup;
        let dt_bio = dt_hw / self.hw_speedup;
        let exp_arg = ((self.v - self.v_rh) / self.delta_t).clamp(-20.0, 20.0);
        let exp_term = self.delta_t * exp_arg.exp();
        let dv = (-(self.v - self.v_rest) + exp_term - self.w + i_ext) / self.tau * dt_bio;
        let dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w * dt_bio;
        let next_v = self.v + dv;
        let next_w = self.w + dw;
        if !dt_hw.is_finite()
            || !dt_bio.is_finite()
            || !exp_term.is_finite()
            || !dv.is_finite()
            || !dw.is_finite()
            || !next_v.is_finite()
            || !next_w.is_finite()
        {
            return Err(BrainScaleSAdExError::NonFiniteUpdate);
        }
        if next_v >= self.v_threshold {
            let spike_w = next_w + self.b;
            if !spike_w.is_finite() {
                return Err(BrainScaleSAdExError::NonFiniteUpdate);
            }
            self.v = self.v_reset;
            self.w = spike_w;
            return Ok(1);
        }
        self.v = next_v;
        self.w = next_w;
        Ok(0)
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.w = 0.0_f64;
    }
}

pub fn validate_brainscales_adex(state: &BrainScaleSAdExNeuron) -> bool {
    state.v.is_finite()
        && state.w.is_finite()
        && state.v_rest.is_finite()
        && state.v_reset.is_finite()
        && state.v_threshold.is_finite()
        && state.delta_t.is_finite()
        && state.v_rh.is_finite()
        && state.tau.is_finite()
        && state.tau_w.is_finite()
        && state.a.is_finite()
        && state.b.is_finite()
        && state.hw_speedup.is_finite()
        && state.dt.is_finite()
        && state.delta_t > 0.0
        && state.tau > 0.0
        && state.tau_w > 0.0
        && state.hw_speedup > 0.0
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brainscales_adex_new() {
        let state = BrainScaleSAdExNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_brainscales_adex(&state));
    }

    #[test]
    fn test_brainscales_adex_step() {
        let mut state = BrainScaleSAdExNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_brainscales_adex_rejects_invalid_input_without_mutation() {
        let mut state = BrainScaleSAdExNeuron::new();
        let before = (state.v, state.w);
        assert_eq!(
            state.step(f64::INFINITY),
            Err(BrainScaleSAdExError::InvalidInput)
        );
        assert_eq!((state.v, state.w), before);
    }

    #[test]
    fn test_brainscales_adex_rejects_nonfinite_update_without_mutation() {
        let mut state = BrainScaleSAdExNeuron::new();
        state.dt = 1.0e308;
        let before = (state.v, state.w);
        assert_eq!(
            state.step(1.0e308),
            Err(BrainScaleSAdExError::NonFiniteUpdate)
        );
        assert_eq!((state.v, state.w), before);
    }
}
