// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Retained WB plus input-driven NMDA project recurrence

use crate::neurons::biophysical::safe_rate;

#[derive(Clone, Debug)]
pub struct SCWBNMDAMagnesiumBlockNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub s_nmda: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_nmda: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_nmda: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub mg_conc: f64,
    pub tau_rise: f64,
    pub tau_decay: f64,
    pub dt: f64,
    pub v_threshold: f64,
    pub gain: f64,
}

impl SCWBNMDAMagnesiumBlockNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.6,
            n: 0.32,
            s_nmda: 0.0,
            g_na: 35.0,
            g_k: 9.0,
            g_nmda: 0.5,
            g_l: 0.1,
            e_na: 55.0,
            e_k: -90.0,
            e_nmda: 0.0,
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0,
            mg_conc: 1.0,
            tau_rise: 10.0,
            tau_decay: 100.0,
            dt: 0.5,
            v_threshold: -20.0,
            gain: 1.0,
        }
    }

    fn valid(&self) -> bool {
        [
            self.v,
            self.h,
            self.n,
            self.s_nmda,
            self.g_na,
            self.g_k,
            self.g_nmda,
            self.g_l,
            self.e_na,
            self.e_k,
            self.e_nmda,
            self.e_l,
            self.c_m,
            self.phi,
            self.mg_conc,
            self.tau_rise,
            self.tau_decay,
            self.dt,
            self.v_threshold,
            self.gain,
        ]
        .into_iter()
        .all(f64::is_finite)
            && (-100.0..=60.0).contains(&self.v)
            && [self.h, self.n, self.s_nmda]
                .into_iter()
                .all(|x| (0.0..=1.0).contains(&x))
            && (0.0..=200.0).contains(&self.g_na)
            && (0.0..=100.0).contains(&self.g_k)
            && (0.0..=20.0).contains(&self.g_nmda)
            && (0.0..=5.0).contains(&self.g_l)
            && (30.0..=70.0).contains(&self.e_na)
            && (-100.0..=-70.0).contains(&self.e_k)
            && (-10.0..=10.0).contains(&self.e_nmda)
            && (-80.0..=-40.0).contains(&self.e_l)
            && (0.5..=2.0).contains(&self.c_m)
            && (0.5..=10.0).contains(&self.phi)
            && (0.0..=5.0).contains(&self.mg_conc)
            && (0.1..=20.0).contains(&self.tau_rise)
            && (10.0..=500.0).contains(&self.tau_decay)
            && self.dt > 0.0
            && self.dt <= 1.0
            && (-20.0..=20.0).contains(&self.v_threshold)
            && (0.0..=10.0).contains(&self.gain)
    }

    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() {
            return Err("current must be finite");
        }
        if !self.valid() {
            return Err("SC NMDA state and parameters must satisfy the public bounds");
        }
        let mut next = self.clone();
        let input = next.gain * current;
        let sub_dt = next.dt / 50.0;
        let drive = if input > 0.0 {
            input / (input + 5.0)
        } else {
            0.0
        };
        let tau = if drive > next.s_nmda {
            next.tau_rise
        } else {
            next.tau_decay
        };
        next.s_nmda = (next.s_nmda + next.dt * (drive - next.s_nmda) / tau).clamp(0.0, 1.0);
        let mut fired = 0;
        for _ in 0..50 {
            let v = next.v;
            let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * (-(v + 60.0) / 18.0).exp();
            let m_inf = alpha_m / (alpha_m + beta_m);
            let alpha_h = 0.07 * (-(v + 58.0) / 20.0).exp();
            let beta_h = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());
            let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * (-(v + 44.0) / 80.0).exp();
            let block = 1.0 / (1.0 + (next.mg_conc / 3.57) * (-0.062 * v).exp());
            next.h += sub_dt * next.phi * (alpha_h * (1.0 - next.h) - beta_h * next.h);
            next.n += sub_dt * next.phi * (alpha_n * (1.0 - next.n) - beta_n * next.n);
            let i_na = next.g_na * m_inf.powi(3) * next.h * (v - next.e_na);
            let i_k = next.g_k * next.n.powi(4) * (v - next.e_k);
            let i_nmda = next.g_nmda * next.s_nmda * block * (v - next.e_nmda);
            let i_l = next.g_l * (v - next.e_l);
            next.v += sub_dt * (-i_na - i_k - i_nmda - i_l + input) / next.c_m;
            if ![next.v, next.h, next.n].into_iter().all(f64::is_finite) {
                return Err("SC NMDA candidate state became non-finite");
            }
            if next.v >= next.v_threshold {
                fired = 1;
                next.v = -65.0;
            }
        }
        next.v = next.v.clamp(-100.0, 60.0);
        next.h = next.h.clamp(0.0, 1.0);
        next.n = next.n.clamp(0.0, 1.0);
        *self = next;
        Ok(fired)
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }
    pub fn reset(&mut self) {
        self.v = -65.0;
        self.h = 0.6;
        self.n = 0.32;
        self.s_nmda = 0.0;
    }
}

impl Default for SCWBNMDAMagnesiumBlockNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_anchor() {
        let mut state = SCWBNMDAMagnesiumBlockNeuron::new();
        assert_eq!(state.try_step(5.0), Ok(0));
        assert!((state.v - -63.155_663_780_395_78).abs() < 1.0e-12);
        assert!((state.s_nmda - 0.025).abs() < 1.0e-15);
    }
}
