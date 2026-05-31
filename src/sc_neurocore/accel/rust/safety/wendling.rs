// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for wendling

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct WendlingNeuron {
    pub y0: f64,
    pub y5: f64,
    pub y1: f64,
    pub y6: f64,
    pub y2: f64,
    pub y7: f64,
    pub y3: f64,
    pub y8: f64,
    pub y4: f64,
    pub y9: f64,
    pub a_exc: f64,
    pub b_fast: f64,
    pub g_slow: f64,
    pub a_rate: f64,
    pub b_rate: f64,
    pub g_rate: f64,
    pub c: f64,
    pub e0: f64,
    pub v0: f64,
    pub r: f64,
    pub dt: f64,
}

impl WendlingNeuron {
    pub fn new() -> Self {
        Self {
            y0: 0.0_f64,
            y5: 0.0_f64,
            y1: 0.0_f64,
            y6: 0.0_f64,
            y2: 0.0_f64,
            y7: 0.0_f64,
            y3: 0.0_f64,
            y8: 0.0_f64,
            y4: 0.0_f64,
            y9: 0.0_f64,
            a_exc: 3.25_f64,
            b_fast: 22.0_f64,
            g_slow: 10.0_f64,
            a_rate: 100.0_f64,
            b_rate: 500.0_f64,
            g_rate: 20.0_f64,
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
        if !i_ext.is_finite() || !validate_wendling(self) {
            return -1;
        }

        let sig_1_2_3_4 = self._sigmoid(self.y1 - self.y2 - self.y3);
        let sig_0 = self._sigmoid(self.c * 0.8 * self.y0);
        let sig_fast = self._sigmoid(self.c * 0.25 * self.y0);
        let sig_slow = self._sigmoid(self.c * 0.1 * self.y0);

        let dy0 = self.y5;
        let dy5 = self.a_exc * self.a_rate * sig_1_2_3_4
            - 2.0 * self.a_rate * self.y5
            - self.a_rate.powi(2) * self.y0;
        let dy1 = self.y6;
        let dy6 = self.a_exc * self.a_rate * (i_ext + self.c * 0.8 * sig_0)
            - 2.0 * self.a_rate * self.y6
            - self.a_rate.powi(2) * self.y1;
        let dy2 = self.y7;
        let dy7 = self.b_fast * self.b_rate * self.c * 0.25 * sig_fast
            - 2.0 * self.b_rate * self.y7
            - self.b_rate.powi(2) * self.y2;
        let dy3 = self.y8;
        let dy8 = self.g_slow * self.g_rate * self.c * 0.1 * sig_slow
            - 2.0 * self.g_rate * self.y8
            - self.g_rate.powi(2) * self.y3;

        let next = Self {
            y0: self.y0 + dy0 * self.dt,
            y5: self.y5 + dy5 * self.dt,
            y1: self.y1 + dy1 * self.dt,
            y6: self.y6 + dy6 * self.dt,
            y2: self.y2 + dy2 * self.dt,
            y7: self.y7 + dy7 * self.dt,
            y3: self.y3 + dy3 * self.dt,
            y8: self.y8 + dy8 * self.dt,
            y4: self.y4,
            y9: self.y9,
            a_exc: self.a_exc,
            b_fast: self.b_fast,
            g_slow: self.g_slow,
            a_rate: self.a_rate,
            b_rate: self.b_rate,
            g_rate: self.g_rate,
            c: self.c,
            e0: self.e0,
            v0: self.v0,
            r: self.r,
            dt: self.dt,
        };
        if !validate_wendling(&next) {
            return -1;
        }
        *self = next;
        0
    }

    pub fn reset(&mut self) {
        // self.y0 = self.y1 = self.y2 = self.y3 = 0.0
        // self.y5 = self.y6 = self.y7 = self.y8 = 0.0
        self.y0 = 0.0_f64;
        self.y5 = 0.0_f64;
        self.y1 = 0.0_f64;
        self.y6 = 0.0_f64;
        self.y2 = 0.0_f64;
        self.y7 = 0.0_f64;
        self.y3 = 0.0_f64;
        self.y8 = 0.0_f64;
        self.y4 = 0.0_f64;
        self.y9 = 0.0_f64;
    }
}

pub fn validate_wendling(state: &WendlingNeuron) -> bool {
    [
        state.y0,
        state.y5,
        state.y1,
        state.y6,
        state.y2,
        state.y7,
        state.y3,
        state.y8,
        state.y4,
        state.y9,
        state.a_exc,
        state.b_fast,
        state.g_slow,
        state.a_rate,
        state.b_rate,
        state.g_rate,
        state.c,
        state.e0,
        state.v0,
        state.r,
        state.dt,
    ]
    .iter()
    .all(|value| value.is_finite())
        && state.a_exc > 0.0
        && state.b_fast > 0.0
        && state.g_slow > 0.0
        && state.a_rate > 0.0
        && state.b_rate > 0.0
        && state.g_rate > 0.0
        && state.c >= 0.0
        && state.e0 > 0.0
        && state.r > 0.0
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wendling_new() {
        let state = WendlingNeuron::new();
        assert!(validate_wendling(&state));
    }

    #[test]
    fn test_wendling_step() {
        let mut state = WendlingNeuron::new();
        let spike = state.step(10.0);
        assert_eq!(spike, 0);
        assert!(state.y6 > 0.0);
    }

    #[test]
    fn test_wendling_rejects_nonfinite_input_without_mutation() {
        let mut state = WendlingNeuron::new();
        let before = state.clone();
        assert_eq!(state.step(f64::NAN), -1);
        assert_eq!(state.y0, before.y0);
        assert_eq!(state.y6, before.y6);
    }
}
