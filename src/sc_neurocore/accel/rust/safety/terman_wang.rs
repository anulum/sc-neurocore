// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for terman_wang

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct TermanWangOscillator {
    pub v: f64,
    pub w: f64,
    pub alpha: f64,
    pub beta: f64,
    pub epsilon: f64,
    pub rho: f64,
    pub dt: f64,
    pub v_peak: f64,
}

impl TermanWangOscillator {
    pub fn new() -> Self {
        Self {
            v: -1.5_f64,
            w: -0.5_f64,
            alpha: 3.0_f64,
            beta: 0.2_f64,
            epsilon: 0.02_f64,
            rho: 0.0_f64,
            dt: 0.05_f64,
            v_peak: 1.5_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !validate_terman_wang(self) {
            return Err("invalid Terman-Wang runtime state");
        }
        if !i_ext.is_finite() {
            return Err("invalid Terman-Wang external current");
        }

        let f = 3.0 * self.v - self.v.powi(3) + 2.0;
        let g = self.alpha * (1.0 + (self.v / self.beta).tanh());
        let dv = (f - self.w + i_ext + self.rho) * self.dt;
        let dw = self.epsilon * (g - self.w) * self.dt;
        if !dv.is_finite() || !dw.is_finite() {
            return Err("non-finite Terman-Wang update");
        }

        let v_prev = self.v;
        let next_v = self.v + dv;
        let next_w = self.w + dw;
        if !next_v.is_finite() || !next_w.is_finite() {
            return Err("non-finite Terman-Wang candidate state");
        }

        self.v = next_v;
        self.w = next_w;
        Ok(if self.v >= self.v_peak && v_prev < self.v_peak {
            1
        } else {
            0
        })
    }

    pub fn reset(&mut self) {
        // self.v = -1.5
        // self.w = -0.5
        self.v = -1.5_f64;
        self.w = -0.5_f64;
        self.alpha = 3.0_f64;
        self.beta = 0.2_f64;
        self.epsilon = 0.02_f64;
    }
}

pub fn validate_terman_wang(state: &TermanWangOscillator) -> bool {
    state.v.is_finite()
        && state.w.is_finite()
        && state.alpha.is_finite()
        && state.beta.is_finite()
        && state.beta > 0.0
        && state.epsilon.is_finite()
        && state.epsilon > 0.0
        && state.rho.is_finite()
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.v_peak.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terman_wang_new() {
        let state = TermanWangOscillator::new();
        assert!(state.v.is_finite());
        assert!(validate_terman_wang(&state));
    }

    #[test]
    fn test_terman_wang_step() {
        let mut state = TermanWangOscillator::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_terman_wang_rejects_invalid_runtime_state() {
        let mut state = TermanWangOscillator::new();
        state.v = f64::INFINITY;
        let before = state.clone();
        assert!(state.step(1.0).is_err());
        assert_eq!(state.v, before.v);
        assert_eq!(state.w, before.w);
    }

    #[test]
    fn test_terman_wang_rejects_invalid_current_without_mutation() {
        let mut state = TermanWangOscillator::new();
        let before = state.clone();
        assert!(state.step(f64::NAN).is_err());
        assert_eq!(state.v, before.v);
        assert_eq!(state.w, before.w);
    }

    #[test]
    fn test_terman_wang_rejects_overflow_candidate_without_mutation() {
        let mut state = TermanWangOscillator::new();
        state.v = 1.0e308;
        let before = state.clone();
        assert!(state.step(1.0).is_err());
        assert_eq!(state.v, before.v);
        assert_eq!(state.w, before.w);
    }
}
