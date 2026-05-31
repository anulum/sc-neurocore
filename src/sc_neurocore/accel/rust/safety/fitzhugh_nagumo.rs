// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for fitzhugh_nagumo

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct FitzHughNagumoNeuron {
    pub v: f64,
    pub w: f64,
    pub a: f64,
    pub b: f64,
    pub epsilon: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl FitzHughNagumoNeuron {
    pub fn new() -> Self {
        Self {
            v: -1.0_f64,
            w: -0.5_f64,
            a: 0.7_f64,
            b: 0.8_f64,
            epsilon: 0.08_f64,
            dt: 0.1_f64,
            v_threshold: 1.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !(i_ext.is_finite() && validate_fitzhugh_nagumo(self)) {
            return -1;
        }
        let v_prev = self.v;
        let dv = (self.v - self.v.powi(3) / 3.0 - self.w + i_ext) * self.dt;
        let dw = self.epsilon * (self.v + self.a - self.b * self.w) * self.dt;
        let mut next = self.clone();
        next.v += dv;
        next.w += dw;
        if !validate_fitzhugh_nagumo(&next) {
            return -1;
        }
        *self = next;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        // self.v = -1.0
        // self.w = -0.5
        self.v = -1.0_f64;
        self.w = -0.5_f64;
        self.a = 0.7_f64;
        self.b = 0.8_f64;
        self.epsilon = 0.08_f64;
        self.dt = 0.1_f64;
        self.v_threshold = 1.0_f64;
    }
}

pub fn validate_fitzhugh_nagumo(state: &FitzHughNagumoNeuron) -> bool {
    state.v.is_finite()
        && state.w.is_finite()
        && state.a.is_finite()
        && state.b.is_finite()
        && state.epsilon.is_finite()
        && state.dt.is_finite()
        && state.v_threshold.is_finite()
        && state.b > 0.0
        && state.epsilon > 0.0
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fitzhugh_nagumo_new() {
        let state = FitzHughNagumoNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_fitzhugh_nagumo(&state));
    }

    #[test]
    fn test_fitzhugh_nagumo_step() {
        let mut state = FitzHughNagumoNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_fitzhugh_nagumo_rejects_invalid_without_mutation() {
        let mut state = FitzHughNagumoNeuron::new();
        let before = state.clone();
        assert_eq!(state.step(f64::NAN), -1);
        assert_eq!(state.v, before.v);
        assert_eq!(state.w, before.w);
    }
}
