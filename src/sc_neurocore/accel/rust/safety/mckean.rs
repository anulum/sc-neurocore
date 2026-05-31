// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for mckean

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct McKeanNeuron {
    pub v: f64,
    pub w: f64,
    pub a: f64,
    pub epsilon: f64,
    pub gamma: f64,
    pub dt: f64,
    pub v_peak: f64,
}

impl McKeanNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0_f64,
            w: 0.0_f64,
            a: 0.25_f64,
            epsilon: 0.01_f64,
            gamma: 0.5_f64,
            dt: 0.1_f64,
            v_peak: 0.8_f64,
        }
    }

    pub fn _f(&self, v: f64) -> f64 {
        let mid1 = self.a / 2.0;
        let mid2 = (1.0 + self.a) / 2.0;
        if v < mid1 {
            -v
        } else if v < mid2 {
            v - self.a
        } else {
            1.0 - v
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_mckean(self) || !i_ext.is_finite() {
            return 0;
        }

        let dv = (self._f(self.v) - self.w + i_ext) * self.dt;
        let dw = self.epsilon * (self.v - self.gamma * self.w) * self.dt;
        let v_prev = self.v;
        let new_v = self.v + dv;
        let new_w = self.w + dw;
        if !(new_v.is_finite() && new_w.is_finite()) {
            return 0;
        }
        self.v = new_v;
        self.w = new_w;
        if self.v >= self.v_peak && v_prev < self.v_peak {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        // self.v = 0.0
        // self.w = 0.0
        self.v = 0.0_f64;
        self.w = 0.0_f64;
        self.a = 0.25_f64;
        self.epsilon = 0.01_f64;
        self.gamma = 0.5_f64;
    }
}

pub fn validate_mckean(state: &McKeanNeuron) -> bool {
    state.v.is_finite()
        && state.w.is_finite()
        && state.a.is_finite()
        && state.a > 0.0
        && state.a < 1.0
        && state.epsilon.is_finite()
        && state.epsilon > 0.0
        && state.gamma.is_finite()
        && state.gamma > 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.v_peak.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mckean_new() {
        let state = McKeanNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_mckean(&state));
    }

    #[test]
    fn test_mckean_step() {
        let mut state = McKeanNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_mckean_current_balance() {
        let mut state = McKeanNeuron::new();
        let spike = state.step(0.5);
        assert_eq!(spike, 0);
        assert!((state.v - 0.05).abs() < 1.0e-12);
        assert!(state.w.abs() < 1.0e-12);
    }

    #[test]
    fn test_mckean_invalid_current_preserves_state() {
        let mut state = McKeanNeuron::new();
        let before = (state.v, state.w);
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!((state.v, state.w), before);
    }

    #[test]
    fn test_mckean_overflow_candidate_preserves_state() {
        let mut state = McKeanNeuron::new();
        state.v = 1.0e308;
        state.w = -1.7e308;
        let before = (state.v, state.w);
        assert_eq!(state.step(1.7e308), 0);
        assert_eq!((state.v, state.w), before);
    }
}
