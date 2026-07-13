// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for adex

#[derive(Debug, Clone)]
pub struct AdExNeuron {
    pub v: f64,
    pub w: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub v_rh: f64,
    pub delta_t: f64,
    pub tau: f64,
    pub tau_w: f64,
    pub a: f64,
    pub b: f64,
    pub c_m: f64,
    pub dt: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdExError {
    InvalidInput,
    InvalidState,
    NonFiniteUpdate,
}

impl Default for AdExNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl AdExNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0_f64,
            w: 0.0_f64,
            v_rest: -65.0_f64,
            v_reset: -68.0_f64,
            v_threshold: -50.0_f64,
            v_rh: -55.0_f64,
            delta_t: 2.0_f64,
            tau: 20.0_f64,
            tau_w: 100.0_f64,
            a: 0.5_f64,
            b: 7.0_f64,
            c_m: 200.0_f64,
            dt: 0.1_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, AdExError> {
        if !i_ext.is_finite() {
            return Err(AdExError::InvalidInput);
        }
        if !validate_adex(self) {
            return Err(AdExError::InvalidState);
        }

        let arg = ((self.v - self.v_rh) / self.delta_t).clamp(-20.0, 20.0);
        let exp_term = self.delta_t * arg.exp();
        let dv = (-(self.v - self.v_rest) + exp_term) / self.tau + (-self.w + i_ext) / self.c_m;
        let dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w;
        let next_v = self.v + dv * self.dt;
        let next_w = self.w + dw * self.dt;
        if !exp_term.is_finite()
            || !dv.is_finite()
            || !dw.is_finite()
            || !next_v.is_finite()
            || !next_w.is_finite()
        {
            return Err(AdExError::NonFiniteUpdate);
        }
        if next_v >= self.v_threshold {
            let spike_w = next_w + self.b;
            if !spike_w.is_finite() {
                return Err(AdExError::NonFiniteUpdate);
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

pub fn validate_adex(state: &AdExNeuron) -> bool {
    state.v.is_finite()
        && state.w.is_finite()
        && state.v_rest.is_finite()
        && state.v_reset.is_finite()
        && state.v_threshold.is_finite()
        && state.v_rh.is_finite()
        && state.delta_t.is_finite()
        && state.tau.is_finite()
        && state.tau_w.is_finite()
        && state.a.is_finite()
        && state.b.is_finite()
        && state.c_m.is_finite()
        && state.dt.is_finite()
        && state.delta_t > 0.0
        && state.tau > 0.0
        && state.tau_w > 0.0
        && state.c_m > 0.0
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adex_new() {
        let state = AdExNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_adex(&state));
    }

    #[test]
    fn test_adex_step() {
        let mut state = AdExNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_adex_matches_python_golden_spike_counts() {
        for (current, expected) in [(0.0, 0), (200.0, 4), (500.0, 12)] {
            let mut state = AdExNeuron::new();
            let spikes: i32 = (0..1_000).map(|_| state.step(current).unwrap()).sum();
            assert_eq!(spikes, expected, "current={current}");
        }
    }

    #[test]
    fn test_adex_one_step_matches_independent_euler_reference() {
        let mut state = AdExNeuron::new();
        state.v = -60.0;
        state.w = 3.0;
        let current = 250.0;
        let arg = ((state.v - state.v_rh) / state.delta_t).clamp(-20.0, 20.0);
        let exp_term = state.delta_t * arg.exp();
        let dv =
            (-(state.v - state.v_rest) + exp_term) / state.tau + (-state.w + current) / state.c_m;
        let dw = (state.a * (state.v - state.v_rest) - state.w) / state.tau_w;
        let expected_v = state.v + dv * state.dt;
        let expected_w = state.w + dw * state.dt;

        assert_eq!(state.step(current), Ok(0));
        assert_eq!(state.v, expected_v);
        assert_eq!(state.w, expected_w);
    }

    #[test]
    fn test_adex_rejects_invalid_input_without_mutation() {
        let mut state = AdExNeuron::new();
        let before = (state.v, state.w);
        assert_eq!(state.step(f64::INFINITY), Err(AdExError::InvalidInput));
        assert_eq!((state.v, state.w), before);
    }

    #[test]
    fn test_adex_rejects_nonfinite_update_without_mutation() {
        let mut state = AdExNeuron::new();
        state.dt = 1.0e308;
        let before = (state.v, state.w);
        assert_eq!(state.step(1.0e308), Err(AdExError::NonFiniteUpdate));
        assert_eq!((state.v, state.w), before);
    }

    #[test]
    fn test_adex_rejects_invalid_runtime_state_without_mutation() {
        let mut state = AdExNeuron::new();
        state.w = f64::NAN;
        let before = (state.v, state.w);
        assert_eq!(state.step(0.0), Err(AdExError::InvalidState));
        assert_eq!(state.v, before.0);
        assert!(state.w.is_nan() && before.1.is_nan());
    }

    #[test]
    fn test_adex_reset_preserves_parameters() {
        let mut state = AdExNeuron::new();
        state.v_rest = -63.0;
        state.dt = 0.2;
        state.a = 0.75;
        state.v = -51.0;
        state.w = 9.0;
        state.reset();
        assert_eq!((state.v, state.w), (-63.0, 0.0));
        assert_eq!((state.dt, state.a), (0.2, 0.75));
    }
}
