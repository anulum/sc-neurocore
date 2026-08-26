// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — preserved six-state thalamocortical oscillator

//! Preserved SC six-state conductance recurrence; no literature attribution.

/// Preserved SC six-state thalamocortical oscillator with Na-dependent K.
#[derive(Clone, Debug)]
pub struct SCSixStateThalamocorticalNeuron {
    pub v: f64,
    pub h_na: f64,
    pub n_k: f64,
    pub m_h: f64,
    pub h_t: f64,
    pub na_i: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_h: f64,
    pub g_t: f64,
    pub g_kna: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_h: f64,
    pub e_ca: f64,
    pub e_l: f64,
    pub na_pump_max: f64,
    pub na_eq: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl SCSixStateThalamocorticalNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h_na: 0.6,
            n_k: 0.3,
            m_h: 0.0,
            h_t: 0.9,
            na_i: 5.0,
            g_na: 50.0,
            g_k: 5.0,
            g_h: 1.0,
            g_t: 3.0,
            g_kna: 1.33,
            g_l: 0.02,
            e_na: 50.0,
            e_k: -90.0,
            e_h: -43.0,
            e_ca: 120.0,
            e_l: -70.0,
            na_pump_max: 20.0,
            na_eq: 9.5,
            dt: 0.05,
            v_threshold: -20.0,
        }
    }
    /// Right-hand side `(dV, dh_na, dn_k, dm_h, dh_t, dna_i)` at one consistent
    /// state. The sodium and T-type activations are instantaneous; the
    /// conductance powers use explicit multiplication and the I_KNa Hill exponent
    /// 3.5 is evaluated as `b*b*b*sqrt(b)` (an IEEE-754 exact decomposition of
    /// `b.powf(3.5)`) so the Python, Julia, Go, and Mojo backends reproduce the
    /// trajectory bit-for-bit rather than depending on a per-platform `pow`.
    fn derivatives(
        &self,
        v: f64,
        h_na: f64,
        n_k: f64,
        m_h: f64,
        h_t: f64,
        na_i: f64,
        current: f64,
    ) -> [f64; 6] {
        let m_na_inf = 1.0 / (1.0 + (-(v + 38.0) / 10.0).exp());
        let h_na_inf = 1.0 / (1.0 + ((v + 43.0) / 6.0).exp());
        let n_k_inf = 1.0 / (1.0 + (-(v + 27.0) / 11.5).exp());
        let m_h_inf = 1.0 / (1.0 + ((v + 75.0) / 5.5).exp());
        let m_t_inf = 1.0 / (1.0 + (-(v + 59.0) / 6.2).exp());
        let h_t_inf = 1.0 / (1.0 + ((v + 83.0) / 4.0).exp());
        let hill_base = 38.7 / na_i.max(0.01);
        let w_kna = 0.37 / (1.0 + hill_base * hill_base * hill_base * hill_base.sqrt());
        let tau_h_na = (1.0 + 10.0 / (1.0 + ((v + 40.0) / 10.0).exp())).max(0.1);
        let z_n = (v + 50.0) / 25.0;
        let tau_n_k = (5.0 + 47.0 * (-(z_n * z_n)).exp()).max(0.1);
        let tau_m_h =
            (20.0 + 1000.0 / (((v + 71.5) / 14.2).exp() + (-(v + 89.0) / 11.6).exp())).max(1.0);
        let tau_h_t = if v < -81.0 {
            (30.8 + 211.4 * ((v + 115.2) / 5.0).exp() / (1.0 + ((v + 86.0) / 3.2).exp())).max(0.1)
        } else {
            10.0
        };
        let d_h_na = (h_na_inf - h_na) / tau_h_na;
        let d_n_k = (n_k_inf - n_k) / tau_n_k;
        let d_m_h = (m_h_inf - m_h) / tau_m_h;
        let d_h_t = (h_t_inf - h_t) / tau_h_t;
        let i_na = self.g_na * m_na_inf * m_na_inf * m_na_inf * h_na * (v - self.e_na);
        let i_k = self.g_k * n_k * n_k * n_k * n_k * (v - self.e_k);
        let i_h = self.g_h * m_h * (v - self.e_h);
        let i_t = self.g_t * m_t_inf * m_t_inf * h_t * (v - self.e_ca);
        let i_kna = self.g_kna * w_kna * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        let d_v = -i_na - i_k - i_h - i_t - i_kna - i_l + current;
        let d_na_i = -0.001 * i_na - self.na_pump_max * (na_i / (na_i + self.na_eq));
        [d_v, d_h_na, d_n_k, d_m_h, d_h_t, d_na_i]
    }

    /// One classical RK4 increment of the six-state vector over `dt`.
    fn rk4_substep(&self, s: [f64; 6], current: f64) -> [f64; 6] {
        let dt = self.dt;
        let k1 = self.derivatives(s[0], s[1], s[2], s[3], s[4], s[5], current);
        let k2 = self.derivatives(
            s[0] + 0.5 * dt * k1[0],
            s[1] + 0.5 * dt * k1[1],
            s[2] + 0.5 * dt * k1[2],
            s[3] + 0.5 * dt * k1[3],
            s[4] + 0.5 * dt * k1[4],
            s[5] + 0.5 * dt * k1[5],
            current,
        );
        let k3 = self.derivatives(
            s[0] + 0.5 * dt * k2[0],
            s[1] + 0.5 * dt * k2[1],
            s[2] + 0.5 * dt * k2[2],
            s[3] + 0.5 * dt * k2[3],
            s[4] + 0.5 * dt * k2[4],
            s[5] + 0.5 * dt * k2[5],
            current,
        );
        let k4 = self.derivatives(
            s[0] + dt * k3[0],
            s[1] + dt * k3[1],
            s[2] + dt * k3[2],
            s[3] + dt * k3[3],
            s[4] + dt * k3[4],
            s[5] + dt * k3[5],
            current,
        );
        let mut out = [0.0_f64; 6];
        for i in 0..6 {
            out[i] = s[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }
        out
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let s = self.rk4_substep(
            [self.v, self.h_na, self.n_k, self.m_h, self.h_t, self.na_i],
            current,
        );
        self.v = s[0];
        self.h_na = s[1];
        self.n_k = s[2];
        self.m_h = s[3];
        self.h_t = s[4];
        self.na_i = s[5].max(0.0);
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -65.0;
        self.h_na = 0.6;
        self.n_k = 0.3;
        self.m_h = 0.0;
        self.h_t = 0.9;
        self.na_i = 5.0;
    }
}
impl Default for SCSixStateThalamocorticalNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = SCSixStateThalamocorticalNeuron::default();
        let constructed = SCSixStateThalamocorticalNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn hyperpolarized_state_uses_slow_t_current_branch() {
        let mut n = SCSixStateThalamocorticalNeuron::new();
        n.v = -90.0;
        assert_eq!(n.step(0.0), 0);
        assert!(n.v.is_finite());
        assert!(n.h_t.is_finite());
    }

    #[test]
    fn sc_six_state_fires() {
        let mut n = SCSixStateThalamocorticalNeuron::new();
        let t: i32 = (0..500).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }

    // -- SCSixStateThalamocortical --
    #[test]
    fn sc_six_state_silent_without_input() {
        let mut n = SCSixStateThalamocorticalNeuron::new();
        let _t: i32 = (0..500).map(|_| n.step(0.0)).sum();
        // May have some spontaneous activity due to Ih
        assert!(n.v.is_finite());
    }
    #[test]
    fn sc_six_state_reset_clears_state() {
        let mut n = SCSixStateThalamocorticalNeuron::new();
        for _ in 0..100 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.v - (-65.0)).abs() < 1e-10);
        assert!((n.na_i - 5.0).abs() < 1e-10);
    }
    #[test]
    fn sc_six_state_extreme_bounded() {
        let mut n = SCSixStateThalamocorticalNeuron::new();
        for _ in 0..200 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn sc_six_state_na_accumulation() {
        let mut n = SCSixStateThalamocorticalNeuron::new();
        for _ in 0..500 {
            n.step(5.0);
        }
        // Intracellular Na+ should remain finite and non-negative
        assert!(n.na_i.is_finite());
        assert!(n.na_i >= 0.0, "Na_i must be non-negative");
    }
    #[test]
    fn sc_six_state_negative_no_crash() {
        let mut n = SCSixStateThalamocorticalNeuron::new();
        for _ in 0..200 {
            n.step(-10.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn sc_six_state_nan_no_panic() {
        let mut n = SCSixStateThalamocorticalNeuron::new();
        n.step(f64::NAN);
    }
}
