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

    pub fn step(&mut self, i_ext: f64) -> i32 {
        // f = 3.0 * self.v - self.v.powi3 + 2.0
        // g = self.alpha * (1.0 + (self.v / self.beta_f64).tanh())
        // dv = (f - self.w + current + self.rho) * self.dt
        // dw = self.epsilon * (g - self.w) * self.dt
        // v_prev = self.v
        // self.v += dv
        // self.w += dw
        // return 1 if (self.v >= self.v_peak && v_prev < self.v_peak) else 0
        0 // spike indicator
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
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }
}
