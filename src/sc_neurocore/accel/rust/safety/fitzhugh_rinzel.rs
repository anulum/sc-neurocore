// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for fitzhugh_rinzel

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct FitzHughRinzelNeuron {
    pub v: f64,
    pub w: f64,
    pub y: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub delta: f64,
    pub mu: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl FitzHughRinzelNeuron {
    pub fn new() -> Self {
        Self {
            v: -1.0_f64,
            w: -0.5_f64,
            y: 0.0_f64,
            a: 0.7_f64,
            b: 0.8_f64,
            c: -0.775_f64,
            d: 1.0_f64,
            delta: 0.08_f64,
            mu: 0.0001_f64,
            dt: 0.1_f64,
            v_threshold: 1.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_fitzhugh_rinzel(self) || !i_ext.is_finite() {
            return 0;
        }
        let v_prev = self.v;
        let dv = self.v - self.v.powi(3) / 3.0 - self.w + self.y + i_ext;
        let dw = self.delta * (self.a + self.v - self.b * self.w);
        let dy = self.mu * (self.c - self.v - self.d * self.y);
        if !(dv.is_finite() && dw.is_finite() && dy.is_finite()) {
            return 0;
        }
        let next_v = self.v + dv * self.dt;
        let next_w = self.w + dw * self.dt;
        let next_y = self.y + dy * self.dt;
        if !(next_v.is_finite() && next_w.is_finite() && next_y.is_finite()) {
            return 0;
        }
        self.v = next_v;
        self.w = next_w;
        self.y = next_y;
        i32::from(v_prev < self.v_threshold && self.v >= self.v_threshold)
    }

    pub fn reset(&mut self) {
        // self.v, self.w, self.y = -1.0, -0.5, 0.0
        self.v = -1.0_f64;
        self.w = -0.5_f64;
        self.y = 0.0_f64;
        self.a = 0.7_f64;
        self.b = 0.8_f64;
    }
}

pub fn validate_fitzhugh_rinzel(state: &FitzHughRinzelNeuron) -> bool {
    state.v.is_finite()
        && state.w.is_finite()
        && state.y.is_finite()
        && state.a.is_finite()
        && state.b.is_finite()
        && state.c.is_finite()
        && state.d.is_finite()
        && state.delta.is_finite()
        && state.mu.is_finite()
        && state.dt.is_finite()
        && state.v_threshold.is_finite()
        && state.delta > 0.0
        && state.mu > 0.0
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fitzhugh_rinzel_new() {
        let state = FitzHughRinzelNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_fitzhugh_rinzel(&state));
    }

    #[test]
    fn test_fitzhugh_rinzel_step() {
        let mut state = FitzHughRinzelNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_fitzhugh_rinzel_current_balance() {
        let mut state = FitzHughRinzelNeuron::new();
        state.v = -1.0;
        state.w = 0.2;
        state.y = 0.1;
        let spike = state.step(0.5);
        assert_eq!(spike, 0);
        assert!((state.v - -1.0266666666666666).abs() < 1.0e-12);
        assert!((state.w - 0.19632).abs() < 1.0e-12);
        assert!((state.y - 0.10000125).abs() < 1.0e-12);
    }

    #[test]
    fn test_fitzhugh_rinzel_invalid_current_preserves_state() {
        let mut state = FitzHughRinzelNeuron::new();
        let before = (state.v, state.w, state.y);
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!((state.v, state.w, state.y), before);
    }

    #[test]
    fn test_fitzhugh_rinzel_overflow_candidate_preserves_state() {
        let mut state = FitzHughRinzelNeuron::new();
        state.v = 1.0e155;
        let before = (state.v, state.w, state.y);
        assert_eq!(state.step(0.5), 0);
        assert_eq!((state.v, state.w, state.y), before);
    }
}
