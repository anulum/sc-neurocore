// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Standalone Wang 1999 NMDA-autapse safety mirror

#[derive(Clone, Debug)]
pub struct NMDANeuron {
    pub v: f64,
    pub x_nmda: f64,
    pub s_nmda: f64,
    pub ca: f64,
    pub refractory_remaining: f64,
    pub c_m: f64,
    pub g_l: f64,
    pub v_l: f64,
    pub g_nmda: f64,
    pub e_nmda: f64,
    pub mg_conc: f64,
    pub alpha_x: f64,
    pub tau_x: f64,
    pub alpha_s: f64,
    pub tau_s: f64,
    pub kinetic_scale: f64,
    pub g_ahp: f64,
    pub v_k: f64,
    pub alpha_ca: f64,
    pub tau_ca: f64,
    pub dt: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub refractory_period: f64,
}

impl NMDANeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            x_nmda: 0.0,
            s_nmda: 0.0,
            ca: 0.0,
            refractory_remaining: 0.0,
            c_m: 0.5,
            g_l: 0.025,
            v_l: -70.0,
            g_nmda: 0.1,
            e_nmda: 0.0,
            mg_conc: 1.0,
            alpha_x: 1.0,
            tau_x: 2.0,
            alpha_s: 1.0,
            tau_s: 80.0,
            kinetic_scale: 1.0,
            g_ahp: 0.0,
            v_k: -85.0,
            alpha_ca: 0.2,
            tau_ca: 80.0,
            dt: 0.05,
            v_threshold: -52.0,
            v_reset: -59.0,
            refractory_period: 2.0,
        }
    }

    fn derivatives(&self, v: f64, x: f64, s: f64, ca: f64, current: f64) -> [f64; 4] {
        let block = 1.0 / (1.0 + self.mg_conc * (-0.062 * v).exp() / 3.57);
        let i_l = self.g_l * (v - self.v_l);
        let i_ahp = self.g_ahp * ca * (v - self.v_k);
        let i_nmda = self.g_nmda * s * block * (v - self.e_nmda);
        [
            (-i_l - i_ahp - i_nmda + current) / self.c_m,
            self.kinetic_scale * (-x / self.tau_x),
            self.kinetic_scale * (self.alpha_s * x * (1.0 - s) - s / self.tau_s),
            -ca / self.tau_ca,
        ]
    }

    fn rk2(&self, v: f64, current: f64) -> [f64; 4] {
        let y = [v, self.x_nmda, self.s_nmda, self.ca];
        let k1 = self.derivatives(y[0], y[1], y[2], y[3], current);
        let h = 0.5 * self.dt;
        let m = [
            y[0] + h * k1[0],
            y[1] + h * k1[1],
            y[2] + h * k1[2],
            y[3] + h * k1[3],
        ];
        let k2 = self.derivatives(m[0], m[1], m[2], m[3], current);
        [
            y[0] + self.dt * k2[0],
            y[1] + self.dt * k2[1],
            y[2] + self.dt * k2[2],
            y[3] + self.dt * k2[3],
        ]
    }

    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() {
            return Err("current must be finite");
        }
        if !validate_nmda_neuron(self) {
            return Err("NMDA state and parameters must satisfy the public bounds");
        }
        let held = self.refractory_remaining > 0.0;
        let mut y = self.rk2(if held { self.v_reset } else { self.v }, current);
        let mut refractory = (self.refractory_remaining - self.dt).max(0.0);
        let mut event = 0;
        if held {
            y[0] = self.v_reset;
        } else if y[0] >= self.v_threshold {
            event = 1;
            y[0] = self.v_reset;
            refractory = self.refractory_period;
            y[1] += self.kinetic_scale * self.alpha_x;
            y[3] += self.alpha_ca;
        }
        if !y.into_iter().chain([refractory]).all(f64::is_finite) {
            return Err("NMDA candidate state became non-finite");
        }
        self.v = y[0].clamp(-120.0, 80.0);
        self.x_nmda = y[1].max(0.0);
        self.s_nmda = y[2].clamp(0.0, 1.0);
        self.ca = y[3].max(0.0);
        self.refractory_remaining = refractory;
        Ok(event)
    }

    pub fn reset(&mut self) {
        self.v = self.v_l;
        self.x_nmda = 0.0;
        self.s_nmda = 0.0;
        self.ca = 0.0;
        self.refractory_remaining = 0.0;
    }
}

impl Default for NMDANeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_nmda_neuron(s: &NMDANeuron) -> bool {
    [
        s.v,
        s.x_nmda,
        s.s_nmda,
        s.ca,
        s.refractory_remaining,
        s.c_m,
        s.g_l,
        s.v_l,
        s.g_nmda,
        s.e_nmda,
        s.mg_conc,
        s.alpha_x,
        s.tau_x,
        s.alpha_s,
        s.tau_s,
        s.kinetic_scale,
        s.g_ahp,
        s.v_k,
        s.alpha_ca,
        s.tau_ca,
        s.dt,
        s.v_threshold,
        s.v_reset,
        s.refractory_period,
    ]
    .into_iter()
    .all(f64::is_finite)
        && (-120.0..=80.0).contains(&s.v)
        && s.x_nmda >= 0.0
        && (0.0..=1.0).contains(&s.s_nmda)
        && s.ca >= 0.0
        && (0.0..=s.refractory_period).contains(&s.refractory_remaining)
        && (0.01..=10.0).contains(&s.c_m)
        && (0.0..=1.0).contains(&s.g_l)
        && (-100.0..=-40.0).contains(&s.v_l)
        && (0.0..=2.0).contains(&s.g_nmda)
        && (-10.0..=10.0).contains(&s.e_nmda)
        && (0.0..=5.0).contains(&s.mg_conc)
        && (0.0..=10.0).contains(&s.alpha_x)
        && (0.01..=100.0).contains(&s.tau_x)
        && (0.0..=10.0).contains(&s.alpha_s)
        && (1.0..=1000.0).contains(&s.tau_s)
        && (0.01..=100.0).contains(&s.kinetic_scale)
        && (0.0..=10.0).contains(&s.g_ahp)
        && (-120.0..=-40.0).contains(&s.v_k)
        && (0.0..=10.0).contains(&s.alpha_ca)
        && (1.0..=1000.0).contains(&s.tau_ca)
        && s.dt > 0.0
        && s.dt <= 0.05
        && (-80.0..=-30.0).contains(&s.v_threshold)
        && s.v_reset >= -100.0
        && s.v_reset < s.v_threshold
        && (0.0..=20.0).contains(&s.refractory_period)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn anchor_and_atomic_failure() {
        let mut n = NMDANeuron::new();
        assert_eq!(n.step(0.3), Ok(0));
        assert!((n.v + 69.9700375).abs() < 1e-12);
        let before = n.clone();
        assert!(n.step(f64::NAN).is_err());
        assert_eq!(n.v, before.v);
    }
}
