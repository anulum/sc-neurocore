// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Standalone retained SC WB plus NMDA safety mirror

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
    pub sub_steps: usize,
}

fn safe_rate(a: f64, vhalf: f64, v: f64, k: f64, fallback: f64) -> f64 {
    let d = v + vhalf;
    if d.abs() < 1e-7 {
        fallback
    } else {
        a * d / (1.0 - (-d / k).exp())
    }
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
            sub_steps: 50,
        }
    }
    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() {
            return Err("current must be finite");
        }
        if !valid(self) {
            return Err("SC NMDA state and parameters must satisfy the public bounds");
        }
        let mut q = self.clone();
        let input = q.gain * current;
        let sub_dt = q.dt / q.sub_steps as f64;
        let drive = if input > 0.0 {
            input / (input + 5.0)
        } else {
            0.0
        };
        let tau = if drive > q.s_nmda {
            q.tau_rise
        } else {
            q.tau_decay
        };
        q.s_nmda = (q.s_nmda + q.dt * (drive - q.s_nmda) / tau).clamp(0.0, 1.0);
        let mut event = 0;
        for _ in 0..q.sub_steps {
            let v = q.v;
            let am = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let bm = 4.0 * (-(v + 60.0) / 18.0).exp();
            let mi = am / (am + bm);
            let ah = 0.07 * (-(v + 58.0) / 20.0).exp();
            let bh = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());
            let an = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let bn = 0.125 * (-(v + 44.0) / 80.0).exp();
            let block = 1.0 / (1.0 + (q.mg_conc / 3.57) * (-0.062 * v).exp());
            q.h += sub_dt * q.phi * (ah * (1.0 - q.h) - bh * q.h);
            q.n += sub_dt * q.phi * (an * (1.0 - q.n) - bn * q.n);
            let ina = q.g_na * mi.powi(3) * q.h * (v - q.e_na);
            let ik = q.g_k * q.n.powi(4) * (v - q.e_k);
            let inmda = q.g_nmda * q.s_nmda * block * (v - q.e_nmda);
            let il = q.g_l * (v - q.e_l);
            q.v += sub_dt * (-ina - ik - inmda - il + input) / q.c_m;
            if ![q.v, q.h, q.n].into_iter().all(f64::is_finite) {
                return Err("SC NMDA candidate state became non-finite");
            }
            if q.v >= q.v_threshold {
                event = 1;
                q.v = -65.0
            }
        }
        q.v = q.v.clamp(-100.0, 60.0);
        q.h = q.h.clamp(0.0, 1.0);
        q.n = q.n.clamp(0.0, 1.0);
        *self = q;
        Ok(event)
    }
    pub fn reset(&mut self) {
        self.v = -65.0;
        self.h = 0.6;
        self.n = 0.32;
        self.s_nmda = 0.0
    }
}
impl Default for SCWBNMDAMagnesiumBlockNeuron {
    fn default() -> Self {
        Self::new()
    }
}
pub fn valid(s: &SCWBNMDAMagnesiumBlockNeuron) -> bool {
    [
        s.v,
        s.h,
        s.n,
        s.s_nmda,
        s.g_na,
        s.g_k,
        s.g_nmda,
        s.g_l,
        s.e_na,
        s.e_k,
        s.e_nmda,
        s.e_l,
        s.c_m,
        s.phi,
        s.mg_conc,
        s.tau_rise,
        s.tau_decay,
        s.dt,
        s.v_threshold,
        s.gain,
    ]
    .into_iter()
    .all(f64::is_finite)
        && (-100.0..=60.0).contains(&s.v)
        && [s.h, s.n, s.s_nmda]
            .into_iter()
            .all(|x| (0.0..=1.0).contains(&x))
        && (0.0..=200.0).contains(&s.g_na)
        && (0.0..=100.0).contains(&s.g_k)
        && (0.0..=20.0).contains(&s.g_nmda)
        && (0.0..=5.0).contains(&s.g_l)
        && (30.0..=70.0).contains(&s.e_na)
        && (-100.0..=-70.0).contains(&s.e_k)
        && (-10.0..=10.0).contains(&s.e_nmda)
        && (-80.0..=-40.0).contains(&s.e_l)
        && (0.5..=2.0).contains(&s.c_m)
        && (0.5..=10.0).contains(&s.phi)
        && (0.0..=5.0).contains(&s.mg_conc)
        && (0.1..=20.0).contains(&s.tau_rise)
        && (10.0..=500.0).contains(&s.tau_decay)
        && s.dt > 0.0
        && s.dt <= 1.0
        && (-20.0..=20.0).contains(&s.v_threshold)
        && (0.0..=10.0).contains(&s.gain)
        && (1..=10000).contains(&s.sub_steps)
}
