// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for alpha_motor_neuron

#![allow(dead_code, non_snake_case)]

const EXP_MAX: f64 = 709.0;
const EXP_MIN: f64 = -745.0;

#[derive(Debug, Clone)]
pub struct AlphaMotorNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub m_pic: f64,
    pub h_pic: f64,
    pub ca: f64,
    pub ca_buf: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_pic: f64,
    pub g_ahp: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub tau_ca: f64,
    pub buf_ratio: f64,
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

impl AlphaMotorNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.8,
            n: 0.1,
            m_pic: 0.0,
            h_pic: 1.0,
            ca: 0.0,
            ca_buf: 0.0,
            g_na: 35.0,
            g_k: 9.0,
            g_pic: 0.15,
            g_ahp: 3.0,
            g_l: 0.3,
            e_na: 55.0,
            e_k: -90.0,
            e_ca: 120.0,
            e_l: -65.0,
            c_m: 1.5,
            phi: 4.0,
            tau_ca: 150.0,
            buf_ratio: 0.003,
            dt: 0.01,
            v_threshold: -20.0,
        }
    }

    pub fn validate(&self) -> bool {
        self.v.is_finite()
            && self.e_na.is_finite()
            && self.e_k.is_finite()
            && self.e_ca.is_finite()
            && self.e_l.is_finite()
            && self.v_threshold.is_finite()
            && probability(self.h)
            && probability(self.n)
            && probability(self.m_pic)
            && probability(self.h_pic)
            && self.ca.is_finite()
            && self.ca >= 0.0
            && self.ca_buf.is_finite()
            && self.ca_buf >= 0.0
            && [self.g_na, self.g_k, self.g_pic, self.g_ahp, self.g_l]
                .iter()
                .all(|x| x.is_finite() && *x >= 0.0)
            && [self.c_m, self.phi, self.tau_ca, self.dt]
                .iter()
                .all(|x| x.is_finite() && *x > 0.0)
            && self.buf_ratio.is_finite()
            && (0.0..=1.0).contains(&self.buf_ratio)
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !self.validate() || !i_ext.is_finite() {
            return Err("invalid alpha motor neuron state or input");
        }
        let v_prev = self.v;
        let mut v = self.v;
        let mut h = self.h;
        let mut n = self.n;
        let mut m_pic = self.m_pic;
        let mut h_pic = self.h_pic;
        let mut ca = self.ca;
        let mut ca_buf = self.ca_buf;
        let n_sub = (0.5 / self.dt.max(0.001)) as usize;
        let n_sub = n_sub.max(1);
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
            let m_pic_inf = 1.0 / (1.0 + checked_exp(-(v + 40.0) / 5.0)?);
            let m_pic_next = m_pic + (m_pic_inf - m_pic) / 50.0 * self.dt;
            let h_pic_inf = 1.0 / (1.0 + checked_exp((v + 40.0) / 8.0)?);
            let tau_h_pic = 200.0 + 100.0 / (1.0 + ((v + 40.0) / 10.0).powi(2)).max(0.01);
            let h_pic_next = (h_pic + (h_pic_inf - h_pic) / tau_h_pic * self.dt).clamp(0.0, 1.0);
            let i_ca_entry = self.g_pic * m_pic_next * h_pic_next * (v - self.e_ca);
            let ca_influx = if i_ca_entry < 0.0 {
                -i_ca_entry * 0.001
            } else {
                0.0
            };
            let ca_spike = if v > -10.0 { 0.02 } else { 0.0 };
            let ca_next = (ca
                + (-ca / self.tau_ca + (ca_influx + ca_spike) * self.buf_ratio) * self.dt)
                .max(0.0);
            let ca_buf_next = (ca_buf
                + ((ca_influx + ca_spike) * (1.0 - self.buf_ratio) - ca_buf / (self.tau_ca * 5.0))
                    * self.dt)
                .max(0.0);
            let ca_total = ca_next + ca_buf_next * 0.01;
            let ahp_inf = ca_total.powi(2) / (ca_total.powi(2) + 0.25);
            let i_na = self.g_na * m_inf.powi(3) * h_next * (v - self.e_na);
            let i_k = self.g_k * n_next.powi(4) * (v - self.e_k);
            let i_pic = self.g_pic * m_pic_next * h_pic_next * (v - self.e_ca);
            let i_ahp = self.g_ahp * ahp_inf * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);
            let v_next = v + (-i_na - i_k - i_pic - i_ahp - i_l + i_ext) / self.c_m * self.dt;
            if !(v_next.is_finite()
                && probability(h_next)
                && probability(n_next)
                && probability(m_pic_next)
                && probability(h_pic_next)
                && ca_next.is_finite()
                && ca_next >= 0.0
                && ca_buf_next.is_finite()
                && ca_buf_next >= 0.0)
            {
                return Err("invalid alpha motor neuron candidate state");
            }
            v = v_next;
            h = h_next;
            n = n_next;
            m_pic = m_pic_next;
            h_pic = h_pic_next;
            ca = ca_next;
            ca_buf = ca_buf_next;
        }
        self.v = v;
        self.h = h;
        self.n = n;
        self.m_pic = m_pic;
        self.h_pic = h_pic;
        self.ca = ca;
        self.ca_buf = ca_buf;
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
        self.m_pic = 0.0;
        self.h_pic = 1.0;
        self.ca = 0.0;
        self.ca_buf = 0.0;
    }
}

pub fn validate_alpha_motor_neuron(state: &AlphaMotorNeuron) -> bool {
    state.validate()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_motor_neuron_new() {
        let state = AlphaMotorNeuron::new();
        assert_eq!(state.v_threshold, -20.0);
        assert!(validate_alpha_motor_neuron(&state));
    }

    #[test]
    fn test_alpha_motor_neuron_step() {
        let mut state = AlphaMotorNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
        assert!(state.v.is_finite());
        assert!(probability(state.h));
        assert!(probability(state.n));
    }

    #[test]
    fn test_alpha_motor_neuron_rejects_invalid_runtime_state() {
        let mut state = AlphaMotorNeuron::new();
        state.h = -0.1;
        let before = state.v;
        assert!(state.step(1.0).is_err());
        assert_eq!(state.v, before);
    }
}
