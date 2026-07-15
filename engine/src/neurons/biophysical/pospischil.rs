// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Pospischil Neuron Model

//! Pospischil minimal conductance model for cortical cell types.

/// Pospischil — minimal HH for 5 cortical cell types. Pospischil et al. 2008.
#[derive(Clone, Debug)]
pub struct PospischilNeuron {
    pub v: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
    pub p: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_m: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub vt: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl PospischilNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            m: 0.05,
            h: 0.6,
            n: 0.3,
            p: 0.0,
            g_na: 50.0,
            g_k: 5.0,
            g_m: 0.07,
            g_l: 0.1,
            e_na: 50.0,
            e_k: -90.0,
            e_l: -70.0,
            c_m: 1.0,
            vt: -56.2,
            dt: 0.025,
            v_threshold: -20.0,
        }
    }
    /// Return `[dV, dm, dh, dn, dp]` of the five-state system at one consistent
    /// state. The Traub-Miles activation rates use the closed-form L'Hôpital limit
    /// within `1e-6` of their `x/(exp(±x/k)-1)` removable singularities, matching
    /// the Python/Julia/Go/Mojo kernels.
    fn derivatives(&self, v: f64, m: f64, h: f64, n: f64, p: f64, current: f64) -> [f64; 5] {
        let dv_vt = v - self.vt;
        let x_m = dv_vt - 13.0;
        let am = if x_m.abs() < 1e-6 {
            1.28
        } else {
            -0.32 * x_m / ((-(x_m) / 4.0).exp() - 1.0)
        };
        let x_bm = dv_vt - 40.0;
        let bm = if x_bm.abs() < 1e-6 {
            1.4
        } else {
            0.28 * x_bm / ((x_bm / 5.0).exp() - 1.0)
        };
        let ah = 0.128 * (-(dv_vt - 17.0) / 18.0).exp();
        let bh = 4.0 / (1.0 + (-(dv_vt - 40.0) / 5.0).exp());
        let x_n = dv_vt - 15.0;
        let an = if x_n.abs() < 1e-6 {
            0.16
        } else {
            -0.032 * x_n / ((-(x_n) / 5.0).exp() - 1.0)
        };
        let bn = 0.5 * (-(dv_vt - 10.0) / 40.0).exp();
        let p_inf = 1.0 / (1.0 + (-(v + 35.0) / 10.0).exp());
        let tau_p = 608.0 / (3.3 * ((v + 35.0) / 20.0).exp() + (-(v + 35.0) / 20.0).exp());
        let dm = am * (1.0 - m) - bm * m;
        let dh = ah * (1.0 - h) - bh * h;
        let dn = an * (1.0 - n) - bn * n;
        let dp = (p_inf - p) / tau_p;
        let i_na = self.g_na * m * m * m * h * (v - self.e_na);
        let i_k = self.g_k * n * n * n * n * (v - self.e_k);
        let i_m = self.g_m * p * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        let dv = (-i_na - i_k - i_m - i_l + current) / self.c_m;
        [dv, dm, dh, dn, dp]
    }

    /// Return one classical RK4 increment of `[V, m, h, n, p]`, holding `current`
    /// constant across the four stages.
    fn rk4_substep(&self, s: [f64; 5], current: f64) -> [f64; 5] {
        let dt = self.dt;
        let k1 = self.derivatives(s[0], s[1], s[2], s[3], s[4], current);
        let k2 = self.derivatives(
            s[0] + 0.5 * dt * k1[0],
            s[1] + 0.5 * dt * k1[1],
            s[2] + 0.5 * dt * k1[2],
            s[3] + 0.5 * dt * k1[3],
            s[4] + 0.5 * dt * k1[4],
            current,
        );
        let k3 = self.derivatives(
            s[0] + 0.5 * dt * k2[0],
            s[1] + 0.5 * dt * k2[1],
            s[2] + 0.5 * dt * k2[2],
            s[3] + 0.5 * dt * k2[3],
            s[4] + 0.5 * dt * k2[4],
            current,
        );
        let k4 = self.derivatives(
            s[0] + dt * k3[0],
            s[1] + dt * k3[1],
            s[2] + dt * k3[2],
            s[3] + dt * k3[3],
            s[4] + dt * k3[4],
            current,
        );
        let mut out = [0.0_f64; 5];
        for i in 0..5 {
            out[i] = s[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }
        out
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let mut s = [self.v, self.m, self.h, self.n, self.p];
        for _ in 0..4 {
            s = self.rk4_substep(s, current);
        }
        self.v = s[0];
        self.m = s[1];
        self.h = s[2];
        self.n = s[3];
        self.p = s[4];
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -70.0;
        self.m = 0.05;
        self.h = 0.6;
        self.n = 0.3;
        self.p = 0.0;
    }
}
impl Default for PospischilNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = PospischilNeuron::default();
        let constructed = PospischilNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn removable_rate_singularities_use_finite_limits() {
        let n = PospischilNeuron::new();
        for voltage in [n.vt + 13.0, n.vt + 40.0, n.vt + 15.0] {
            assert!(n
                .derivatives(voltage, n.m, n.h, n.n, n.p, 0.0)
                .iter()
                .all(|value| value.is_finite()));
        }
    }

    #[test]
    fn pospischil_fires() {
        let mut n = PospischilNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }

    // -- Pospischil --
    #[test]
    fn pospischil_silent_without_input() {
        let mut n = PospischilNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn pospischil_reset_clears_state() {
        let mut n = PospischilNeuron::new();
        for _ in 0..100 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.v - (-70.0)).abs() < 1e-10);
    }
    #[test]
    fn pospischil_moderate_input_stable() {
        let mut n = PospischilNeuron::new();
        for _ in 0..200 {
            n.step(10.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn pospischil_m_current_present() {
        let mut n = PospischilNeuron::new();
        for _ in 0..200 {
            n.step(5.0);
        }
        assert!(n.p > 0.0, "M-current (p) should activate during spiking");
    }
    #[test]
    fn pospischil_negative_no_crash() {
        let mut n = PospischilNeuron::new();
        for _ in 0..200 {
            n.step(-10.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn pospischil_nan_no_panic() {
        let mut n = PospischilNeuron::new();
        n.step(f64::NAN);
    }
}
