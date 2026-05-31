// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for bk_neuron

#![allow(dead_code, non_snake_case)]

const EXP_MAX: f64 = 709.0;
const EXP_MIN: f64 = -745.0;

#[derive(Debug, Clone)]
pub struct BKNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub ca: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_bk: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub tau_ca: f64,
    pub dt: f64,
    pub v_threshold: f64,
    pub gain: f64,
    pub sub_steps: usize,
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

impl BKNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.6,
            n: 0.32,
            ca: 0.0,
            g_na: 35.0,
            g_k: 9.0,
            g_bk: 3.0,
            g_l: 0.1,
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0,
            tau_ca: 50.0,
            dt: 0.5,
            v_threshold: -20.0,
            gain: 1.0,
            sub_steps: 50,
        }
    }

    pub fn validate(&self) -> bool {
        [
            self.v,
            self.e_na,
            self.e_k,
            self.e_l,
            self.v_threshold,
            self.gain,
        ]
        .iter()
        .all(|x| x.is_finite())
            && probability(self.h)
            && probability(self.n)
            && self.ca.is_finite()
            && self.ca >= 0.0
            && [self.g_na, self.g_k, self.g_bk, self.g_l]
                .iter()
                .all(|x| x.is_finite() && *x >= 0.0)
            && [self.c_m, self.phi, self.tau_ca, self.dt]
                .iter()
                .all(|x| x.is_finite() && *x > 0.0)
            && self.sub_steps > 0
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !self.validate() || !i_ext.is_finite() {
            return Err("invalid BK neuron state or input");
        }
        let inp = self.gain * i_ext;
        if !inp.is_finite() {
            return Err("invalid BK input drive");
        }
        let sub_dt = self.dt / self.sub_steps as f64;
        if !sub_dt.is_finite() || sub_dt <= 0.0 {
            return Err("invalid BK substep");
        }
        let mut v = self.v;
        let mut h = self.h;
        let mut n = self.n;
        let mut ca = self.ca;
        let mut fired = 0;
        for _ in 0..self.sub_steps {
            let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0)?;
            let beta_m = 4.0 * checked_exp(-(v + 60.0) / 18.0)?;
            let m_inf = alpha_m / (alpha_m + beta_m);
            let alpha_h = 0.07 * checked_exp(-(v + 58.0) / 20.0)?;
            let beta_h = 1.0 / (1.0 + checked_exp(-(v + 28.0) / 10.0)?);
            let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1)?;
            let beta_n = 0.125 * checked_exp(-(v + 44.0) / 80.0)?;
            let ca_next_decay = (ca + sub_dt * (-ca / self.tau_ca)).max(0.0);
            if !ca_next_decay.is_finite() {
                return Err("invalid BK calcium candidate");
            }
            let denom = ca_next_decay + 0.5;
            if !denom.is_finite() || denom <= 0.0 {
                return Err("invalid BK calcium denominator");
            }
            let v_half_bk = 10.0 - 30.0 * (ca_next_decay / denom);
            let bk_inf = 1.0 / (1.0 + checked_exp(-(v - v_half_bk) / 15.0)?);
            if !probability(bk_inf) {
                return Err("invalid BK activation candidate");
            }
            let h_next = h + sub_dt * self.phi * (alpha_h * (1.0 - h) - beta_h * h);
            let n_next = n + sub_dt * self.phi * (alpha_n * (1.0 - n) - beta_n * n);
            let i_na = self.g_na * m_inf.powi(3) * h_next * (v - self.e_na);
            let i_k = self.g_k * n_next.powi(4) * (v - self.e_k);
            let i_bk = self.g_bk * bk_inf * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);
            let dv = (-i_na - i_k - i_bk - i_l + inp) / self.c_m;
            let mut v_next = v + sub_dt * dv;
            let mut ca_next = ca_next_decay;
            if v_next >= self.v_threshold {
                fired = 1;
                v_next = -65.0;
                ca_next += 0.3;
            }
            if !(v_next.is_finite()
                && (-100.0..=60.0).contains(&v_next)
                && probability(h_next)
                && probability(n_next)
                && ca_next.is_finite()
                && ca_next >= 0.0)
            {
                return Err("invalid BK candidate state");
            }
            v = v_next;
            h = h_next;
            n = n_next;
            ca = ca_next;
        }
        self.v = v;
        self.h = h;
        self.n = n;
        self.ca = ca;
        Ok(fired)
    }

    pub fn reset(&mut self) {
        self.v = -65.0;
        self.h = 0.6;
        self.n = 0.32;
        self.ca = 0.0;
    }
}

pub fn validate_bk_neuron(state: &BKNeuron) -> bool {
    state.validate()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bk_neuron_new() {
        let state = BKNeuron::new();
        assert!(validate_bk_neuron(&state));
    }

    #[test]
    fn test_bk_neuron_step() {
        let mut state = BKNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
        assert!(state.v.is_finite());
        assert!(state.ca >= 0.0);
    }

    #[test]
    fn test_bk_neuron_rejects_invalid_runtime_state() {
        let mut state = BKNeuron::new();
        state.ca = f64::INFINITY;
        let before = state.v;
        assert!(state.step(1.0).is_err());
        assert_eq!(state.v, before);
    }
}
