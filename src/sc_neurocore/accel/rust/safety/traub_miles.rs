// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for traub_miles

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct TraubMilesNeuron {
    pub v: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl TraubMilesNeuron {
    pub fn new() -> Self {
        Self {
            v: -67.0_f64,
            m: 0.05_f64,
            h: 0.6_f64,
            n: 0.3_f64,
            g_na: 100.0_f64,
            g_k: 80.0_f64,
            g_l: 0.1_f64,
            e_na: 50.0_f64,
            e_k: -100.0_f64,
            e_l: -67.0_f64,
            dt: 0.01_f64,
            v_threshold: -20.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !validate_traub_miles(self) {
            return Err("invalid Traub-Miles runtime state");
        }
        if !i_ext.is_finite() {
            return Err("invalid Traub-Miles external current");
        }

        let v_prev = self.v;
        let mut v = self.v;
        let mut m = self.m;
        let mut h = self.h;
        let mut n = self.n;
        for _ in 0..10 {
            let (am, bm, ah, bh, an, bn) = rates(v)?;
            let next_m = m + (am * (1.0 - m) - bm * m) * self.dt;
            let next_h = h + (ah * (1.0 - h) - bh * h) * self.dt;
            let next_n = n + (an * (1.0 - n) - bn * n) * self.dt;
            if !finite_gate(next_m) || !finite_gate(next_h) || !finite_gate(next_n) {
                return Err("invalid Traub-Miles gate candidate");
            }
            let i_na = self.g_na * next_m.powi(3) * next_h * (v - self.e_na);
            let i_k = self.g_k * next_n.powi(4) * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);
            let next_v = v + (-i_na - i_k - i_l + i_ext) * self.dt;
            if !next_v.is_finite() {
                return Err("invalid Traub-Miles voltage candidate");
            }
            v = next_v;
            m = next_m;
            h = next_h;
            n = next_n;
        }

        self.v = v;
        self.m = m;
        self.h = h;
        self.n = n;
        Ok(if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        })
    }

    pub fn reset(&mut self) {
        // self.v, self.m, self.h, self.n = -67.0, 0.05, 0.6, 0.3
        self.v = -67.0_f64;
        self.m = 0.05_f64;
        self.h = 0.6_f64;
        self.n = 0.3_f64;
        self.g_na = 100.0_f64;
    }
}

pub fn validate_traub_miles(state: &TraubMilesNeuron) -> bool {
    state.v.is_finite()
        && finite_gate(state.m)
        && finite_gate(state.h)
        && finite_gate(state.n)
        && state.g_na.is_finite()
        && state.g_na >= 0.0
        && state.g_k.is_finite()
        && state.g_k >= 0.0
        && state.g_l.is_finite()
        && state.g_l >= 0.0
        && state.e_na.is_finite()
        && state.e_k.is_finite()
        && state.e_l.is_finite()
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.v_threshold.is_finite()
}

fn finite_gate(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn rates(v: f64) -> Result<(f64, f64, f64, f64, f64, f64), &'static str> {
    let d = v + 54.0;
    let am = if d.abs() > 1e-6 {
        0.32 * d / (1.0 - (-d / 4.0).exp())
    } else {
        8.0
    };
    let d2 = v + 27.0;
    let bm = if d2.abs() > 1e-6 {
        0.28 * d2 / ((d2 / 5.0).exp() - 1.0)
    } else {
        5.6
    };
    let ah = 0.128 * (-(v + 50.0) / 18.0).exp();
    let bh = 4.0 / (1.0 + (-(v + 27.0) / 5.0).exp());
    let d3 = v + 52.0;
    let an = if d3.abs() > 1e-6 {
        0.032 * d3 / (1.0 - (-d3 / 5.0).exp())
    } else {
        0.32
    };
    let bn = 0.5 * (-(v + 57.0) / 40.0).exp();
    for rate in [am, bm, ah, bh, an, bn] {
        if !rate.is_finite() || rate < 0.0 {
            return Err("invalid Traub-Miles rate");
        }
    }
    Ok((am, bm, ah, bh, an, bn))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_traub_miles_new() {
        let state = TraubMilesNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_traub_miles(&state));
    }

    #[test]
    fn test_traub_miles_step() {
        let mut state = TraubMilesNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_traub_miles_rejects_invalid_runtime_state() {
        let mut state = TraubMilesNeuron::new();
        state.m = 1.5;
        assert!(state.step(1.0).is_err());
    }
}
