// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for chandelier_neuron

#![allow(dead_code, non_snake_case)]

const EXP_MAX: f64 = 709.0;
const EXP_MIN: f64 = -745.0;

#[derive(Debug, Clone)]
pub struct ChandelierNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub d: f64,
    pub p: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_kv1: f64,
    pub g_kv3: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

fn probability(x: f64) -> bool {
    x.is_finite() && (0.0..=1.0).contains(&x)
}

fn checked_exp(x: f64) -> Result<f64, &'static str> {
    if !x.is_finite() || x > EXP_MAX {
        Err("unstable exponential argument")
    } else if x < EXP_MIN {
        Ok(0.0)
    } else {
        Ok(x.exp())
    }
}

fn safe_rate(a: f64, vhalf: f64, v: f64, k: f64, fallback: f64) -> Result<f64, &'static str> {
    let d = v + vhalf;
    if d.abs() < 1e-7 {
        return Ok(fallback);
    }
    let rate = a * d / (1.0 - checked_exp(-d / k)?);
    if rate.is_finite() {
        Ok(rate)
    } else {
        Err("non-finite rate candidate")
    }
}

impl ChandelierNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.8,
            n: 0.1,
            d: 0.0,
            p: 0.0,
            g_na: 35.0,
            g_k: 9.0,
            g_kv1: 3.0,
            g_kv3: 4.0,
            g_l: 0.1,
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0,
            dt: 0.01,
            v_threshold: -20.0,
        }
    }

    pub fn validate(&self) -> bool {
        [self.v, self.e_na, self.e_k, self.e_l, self.v_threshold]
            .iter()
            .all(|x| x.is_finite())
            && [self.h, self.n, self.d, self.p]
                .iter()
                .all(|x| probability(*x))
            && [self.g_na, self.g_k, self.g_kv1, self.g_kv3, self.g_l]
                .iter()
                .all(|x| x.is_finite() && *x >= 0.0)
            && [self.c_m, self.phi, self.dt]
                .iter()
                .all(|x| x.is_finite() && *x > 0.0)
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !self.validate() || !i_ext.is_finite() {
            return Err("invalid chandelier state or input");
        }
        let v_prev = self.v;
        let n_sub = ((0.5 / self.dt.max(0.001)) as usize).max(1);
        let mut v = self.v;
        let mut h = self.h;
        let mut n = self.n;
        let mut d_gate = self.d;
        let mut p_gate = self.p;
        for _ in 0..n_sub {
            let am = safe_rate(0.1, 35.0, v, 10.0, 1.0)?;
            let bm = 4.0 * checked_exp(-(v + 60.0) / 18.0)?;
            let m_inf = am / (am + bm);
            let ah = 0.07 * checked_exp(-(v + 58.0) / 20.0)?;
            let bh = 1.0 / (1.0 + checked_exp(-(v + 28.0) / 10.0)?);
            let an = safe_rate(0.01, 34.0, v, 10.0, 0.1)?;
            let bn = 0.125 * checked_exp(-(v + 44.0) / 80.0)?;
            let h_next = h + self.phi * (ah * (1.0 - h) - bh * h) * self.dt;
            let n_next = n + self.phi * (an * (1.0 - n) - bn * n) * self.dt;
            let d_inf = 1.0 / (1.0 + checked_exp(-(v + 50.0) / 10.0)?);
            let d_next = d_gate + (d_inf - d_gate) / 150.0 * self.dt;
            let p_inf = 1.0 / (1.0 + checked_exp(-(v + 10.0) / 10.0)?);
            let p_next = p_gate + self.phi * (p_inf - p_gate) * self.dt;
            let i_na = self.g_na * m_inf.powi(3) * h_next * (v - self.e_na);
            let i_k = self.g_k * n_next.powi(4) * (v - self.e_k);
            let i_kv1 = self.g_kv1 * d_next.powi(4) * (v - self.e_k);
            let i_kv3 = self.g_kv3 * p_next * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);
            let v_next = v + (-i_na - i_k - i_kv1 - i_kv3 - i_l + i_ext) / self.c_m * self.dt;
            if !(v_next.is_finite()
                && (-100.0..=60.0).contains(&v_next)
                && probability(h_next)
                && probability(n_next)
                && probability(d_next)
                && probability(p_next))
            {
                return Err("invalid chandelier candidate state");
            }
            v = v_next;
            h = h_next;
            n = n_next;
            d_gate = d_next;
            p_gate = p_next;
        }
        self.v = v;
        self.h = h;
        self.n = n;
        self.d = d_gate;
        self.p = p_gate;
        Ok(if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        })
    }

    pub fn reset(&mut self) {
        self.v = -65.0;
        self.h = 0.8;
        self.n = 0.1;
        self.d = 0.0;
        self.p = 0.0;
    }
}

pub fn validate_chandelier_neuron(state: &ChandelierNeuron) -> bool {
    state.validate()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chandelier_neuron_new() {
        let state = ChandelierNeuron::new();
        assert!(validate_chandelier_neuron(&state));
    }

    #[test]
    fn test_chandelier_neuron_step() {
        let mut state = ChandelierNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
        assert!(state.v.is_finite());
        assert!(probability(state.d));
        assert!(probability(state.p));
    }

    #[test]
    fn test_chandelier_neuron_rejects_invalid_runtime_state() {
        let mut state = ChandelierNeuron::new();
        state.p = 1.5;
        let before = state.v;
        assert!(state.step(1.0).is_err());
        assert_eq!(state.v, before);
    }
}
