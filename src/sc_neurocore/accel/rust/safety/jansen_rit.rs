// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for jansen_rit

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct JansenRitUnit {
    pub y0: f64,
    pub y3: f64,
    pub y1: f64,
    pub y4: f64,
    pub y2: f64,
    pub y5: f64,
    pub a_exc: f64,
    pub b_exc: f64,
    pub a_rate: f64,
    pub b_rate: f64,
    pub c: f64,
    pub e0: f64,
    pub v0: f64,
    pub r: f64,
    pub dt: f64,
}

impl JansenRitUnit {
    pub fn new() -> Self {
        Self {
            y0: 0.0_f64,
            y3: 0.0_f64,
            y1: 0.0_f64,
            y4: 0.0_f64,
            y2: 0.0_f64,
            y5: 0.0_f64,
            a_exc: 3.25_f64,
            b_exc: 22.0_f64,
            a_rate: 100.0_f64,
            b_rate: 50.0_f64,
            c: 135.0_f64,
            e0: 2.5_f64,
            v0: 6.0_f64,
            r: 0.56_f64,
            dt: 0.001_f64,
        }
    }

    pub fn _sigmoid(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NAN;
        }
        let exponent = self.r * (self.v0 - x);
        if exponent >= 0.0 {
            let exp_neg = (-exponent).exp();
            2.0 * self.e0 * exp_neg / (1.0 + exp_neg)
        } else {
            2.0 * self.e0 / (1.0 + exponent.exp())
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !i_ext.is_finite() || !validate_jansen_rit(self) {
            return -1;
        }
        let s1 = self._sigmoid(self.y1 - self.y2);
        let s0 = self._sigmoid(self.c * 0.8 * self.y0);
        let s2 = self._sigmoid(self.c * 0.25 * self.y0);
        let dy0 = self.y3;
        let dy3 = self.a_exc * self.a_rate * s1
            - 2.0 * self.a_rate * self.y3
            - self.a_rate.powi(2) * self.y0;
        let dy1 = self.y4;
        let dy4 = self.a_exc * self.a_rate * (i_ext + self.c * 0.8 * s0)
            - 2.0 * self.a_rate * self.y4
            - self.a_rate.powi(2) * self.y1;
        let dy2 = self.y5;
        let dy5 = self.b_exc * self.b_rate * self.c * 0.25 * s2
            - 2.0 * self.b_rate * self.y5
            - self.b_rate.powi(2) * self.y2;

        let next = Self {
            y0: self.y0 + dy0 * self.dt,
            y3: self.y3 + dy3 * self.dt,
            y1: self.y1 + dy1 * self.dt,
            y4: self.y4 + dy4 * self.dt,
            y2: self.y2 + dy2 * self.dt,
            y5: self.y5 + dy5 * self.dt,
            a_exc: self.a_exc,
            b_exc: self.b_exc,
            a_rate: self.a_rate,
            b_rate: self.b_rate,
            c: self.c,
            e0: self.e0,
            v0: self.v0,
            r: self.r,
            dt: self.dt,
        };
        if !validate_jansen_rit(&next) {
            return -1;
        }
        *self = next;
        0
    }

    pub fn reset(&mut self) {
        // self.y0 = self.y1 = self.y2 = self.y3 = self.y4 = self.y5 = 0.0
        self.y0 = 0.0_f64;
        self.y3 = 0.0_f64;
        self.y1 = 0.0_f64;
        self.y4 = 0.0_f64;
        self.y2 = 0.0_f64;
        self.y5 = 0.0_f64;
    }
}

pub fn validate_jansen_rit(state: &JansenRitUnit) -> bool {
    [
        state.y0,
        state.y3,
        state.y1,
        state.y4,
        state.y2,
        state.y5,
        state.a_exc,
        state.b_exc,
        state.a_rate,
        state.b_rate,
        state.c,
        state.e0,
        state.v0,
        state.r,
        state.dt,
    ]
    .iter()
    .all(|value| value.is_finite())
        && state.a_exc > 0.0
        && state.b_exc > 0.0
        && state.a_rate > 0.0
        && state.b_rate > 0.0
        && state.c >= 0.0
        && state.e0 > 0.0
        && state.r > 0.0
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jansen_rit_new() {
        let state = JansenRitUnit::new();
        assert!(validate_jansen_rit(&state));
    }

    #[test]
    fn test_jansen_rit_step() {
        let mut state = JansenRitUnit::new();
        let spike = state.step(10.0);
        assert_eq!(spike, 0);
        assert!(state.y4 > 0.0);
    }

    #[test]
    fn test_jansen_rit_rejects_nonfinite_input_without_mutation() {
        let mut state = JansenRitUnit::new();
        let before = state.clone();
        assert_eq!(state.step(f64::NAN), -1);
        assert_eq!(state.y0, before.y0);
        assert_eq!(state.y4, before.y4);
    }
}
