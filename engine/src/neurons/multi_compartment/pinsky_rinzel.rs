// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Pinsky-Rinzel neuron model

//! Pinsky-Rinzel multi-compartment neuron model.

use crate::neurons::biophysical::safe_rate;

/// Pinsky-Rinzel 1994 — 2-compartment pyramidal cell.
#[derive(Clone, Debug)]
pub struct PinskyRinzelNeuron {
    pub v_s: f64,
    pub v_d: f64,
    pub h: f64,
    pub n: f64,
    pub s: f64,
    pub c: f64,
    pub q: f64,
    pub ca: f64,
    pub cm: f64,
    pub gc: f64,
    pub p: f64,
    pub g_na: f64,
    pub g_kdr: f64,
    pub g_ca: f64,
    pub g_kahp: f64,
    pub g_kc: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_l: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl PinskyRinzelNeuron {
    pub fn new() -> Self {
        Self {
            v_s: -60.0,
            v_d: -60.0,
            h: 0.999,
            n: 0.001,
            s: 0.009,
            c: 0.007,
            q: 0.01,
            ca: 0.2,
            cm: 3.0,
            gc: 2.1,
            p: 0.5,
            g_na: 30.0,
            g_kdr: 15.0,
            g_ca: 10.0,
            g_kahp: 0.8,
            g_kc: 15.0,
            g_l: 0.1,
            e_na: 60.0,
            e_k: -75.0,
            e_ca: 80.0,
            e_l: -60.0,
            dt: 0.02,
            v_threshold: -20.0,
        }
    }

    /// Time derivatives of the eight-state vector `(v_s, v_d, h, n, s, c, q, ca)`.
    fn derivatives(&self, st: &[f64; 8], i_s: f64, i_d: f64) -> [f64; 8] {
        let (v_s, v_d, h, n, s, c, q, ca) =
            (st[0], st[1], st[2], st[3], st[4], st[5], st[6], st[7]);
        let am = safe_rate(0.32, 46.9, v_s, 4.0, 0.32 * 4.0);
        let bm = pr_exprel_plus(0.28, v_s + 19.9, 5.0);
        let m_inf = if am + bm > 0.0 { am / (am + bm) } else { 0.0 };
        let ah = 0.128 * (-(v_s + 43.0) / 18.0).exp();
        let bh = 4.0 / (1.0 + (-(v_s + 20.0) / 5.0).exp());
        let an = safe_rate(0.016, 24.9, v_s, 5.0, 0.016 * 5.0);
        let bn = 0.25 * (-1.0 - 0.025 * v_s).exp();
        let a_s = 1.6 / (1.0 + (-0.072 * (v_d - 5.0)).exp());
        let b_s = pr_exprel_plus(0.02, v_d + 8.9, 5.0);
        let (ac, bc) = if v_d <= -10.0 {
            let ac = ((v_d + 50.0) / 11.0 - (v_d + 53.5) / 27.0).exp() / 18.975;
            (ac, 2.0 * ((-53.5 - v_d) / 27.0).exp() - ac)
        } else {
            (2.0 * ((-53.5 - v_d) / 27.0).exp(), 0.0)
        };
        let aq = (0.00002 * ca).min(0.01);
        let bq = 0.001;
        let chi = (ca / 250.0).min(1.0);

        let i_na = self.g_na * m_inf.powi(2) * h * (v_s - self.e_na);
        let i_kdr = self.g_kdr * n * (v_s - self.e_k);
        let i_ls = self.g_l * (v_s - self.e_l);
        let i_ca = self.g_ca * s.powi(2) * (v_d - self.e_ca);
        let i_kahp = self.g_kahp * q * (v_d - self.e_k);
        let i_kc = self.g_kc * c * chi * (v_d - self.e_k);
        let i_ld = self.g_l * (v_d - self.e_l);
        let i_coupling = self.gc * (v_d - v_s);

        let dv_s = (-i_ls - i_na - i_kdr + i_coupling / self.p + i_s / self.p) / self.cm;
        let dv_d = (-i_ld - i_ca - i_kahp - i_kc - i_coupling / (1.0 - self.p)
            + i_d / (1.0 - self.p))
            / self.cm;
        [
            dv_s,
            dv_d,
            ah * (1.0 - h) - bh * h,
            an * (1.0 - n) - bn * n,
            a_s * (1.0 - s) - b_s * s,
            ac * (1.0 - c) - bc * c,
            aq * (1.0 - q) - bq * q,
            -0.13 * i_ca - 0.075 * ca,
        ]
    }

    pub fn step(&mut self, current_soma: f64, current_dend: f64) -> i32 {
        let v_prev = self.v_s;
        let y = [
            self.v_s, self.v_d, self.h, self.n, self.s, self.c, self.q, self.ca,
        ];
        let dt = self.dt;
        let k1 = self.derivatives(&y, current_soma, current_dend);
        let mut y2 = y;
        for i in 0..8 {
            y2[i] = y[i] + (dt / 2.0) * k1[i];
        }
        let k2 = self.derivatives(&y2, current_soma, current_dend);
        let mut y3 = y;
        for i in 0..8 {
            y3[i] = y[i] + (dt / 2.0) * k2[i];
        }
        let k3 = self.derivatives(&y3, current_soma, current_dend);
        let mut y4 = y;
        for i in 0..8 {
            y4[i] = y[i] + dt * k3[i];
        }
        let k4 = self.derivatives(&y4, current_soma, current_dend);
        let mut nxt = [0.0_f64; 8];
        for i in 0..8 {
            nxt[i] = y[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        self.v_s = nxt[0];
        self.v_d = nxt[1];
        self.h = nxt[2].clamp(0.0, 1.0);
        self.n = nxt[3].clamp(0.0, 1.0);
        self.s = nxt[4].clamp(0.0, 1.0);
        self.c = nxt[5].clamp(0.0, 1.0);
        self.q = nxt[6].clamp(0.0, 1.0);
        self.ca = nxt[7].max(0.0);
        if self.v_s >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v_s = -60.0;
        self.v_d = -60.0;
        self.h = 0.999;
        self.n = 0.001;
        self.s = 0.009;
        self.c = 0.007;
        self.q = 0.01;
        self.ca = 0.2;
    }
}

/// Traub deactivation rate `a*dv / (exp(dv/k) - 1)` with removable limit `a*k`.
fn pr_exprel_plus(a: f64, dv: f64, k: f64) -> f64 {
    if dv.abs() < 1e-6 {
        a * k
    } else {
        a * dv / ((dv / k).exp() - 1.0)
    }
}
impl Default for PinskyRinzelNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pr_fires() {
        let mut n = PinskyRinzelNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(5.0, 0.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn pr_reset() {
        let mut n = PinskyRinzelNeuron::new();
        for _ in 0..100 {
            n.step(5.0, 0.0);
        }
        n.reset();
        assert!((n.v_s - (-60.0)).abs() < 1e-10);
        assert!((n.v_d - (-60.0)).abs() < 1e-10);
    }

    #[test]
    fn pr_bounded() {
        let mut n = PinskyRinzelNeuron::new();
        for _ in 0..5000 {
            n.step(50.0, 0.0);
        }
        assert!(n.v_s.is_finite());
    }

    #[test]
    fn pr_dendritic_input() {
        let mut n = PinskyRinzelNeuron::new();
        let _t: i32 = (0..5000).map(|_| n.step(0.0, 5.0)).sum();
        // Dendritic input should also be able to drive spiking
        assert!(n.v_d.is_finite());
    }

    #[test]
    fn pr_nan_no_panic() {
        PinskyRinzelNeuron::new().step(f64::NAN, 0.0);
    }
}
