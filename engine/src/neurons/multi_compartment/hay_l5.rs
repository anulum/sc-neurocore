// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Hay layer-5 pyramidal neuron model

//! Hay layer-5 pyramidal neuron model.

/// Hay et al. 2011 — Layer 5 thick-tufted pyramidal (3-compartment reduced).
#[derive(Clone, Debug)]
pub struct HayL5PyramidalNeuron {
    pub v_s: f64,
    pub v_t: f64,
    pub v_a: f64,
    pub h_na: f64,
    pub n_k: f64,
    pub m_ca: f64,
    pub h_ca: f64,
    pub m_ih: f64,
    pub ca_a: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_l_s: f64,
    pub g_ca_t: f64,
    pub g_ih: f64,
    pub g_l_t: f64,
    pub g_ca_a: f64,
    pub g_kca: f64,
    pub g_l_a: f64,
    pub g_st: f64,
    pub g_ta: f64,
    pub p_s: f64,
    pub p_t: f64,
    pub p_a: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_ih: f64,
    pub e_l: f64,
    pub ca_decay: f64,
    pub f_ca: f64,
    pub c_m: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl HayL5PyramidalNeuron {
    pub fn new() -> Self {
        Self {
            v_s: -75.0,
            v_t: -75.0,
            v_a: -75.0,
            h_na: 0.9,
            n_k: 0.1,
            m_ca: 0.0,
            h_ca: 1.0,
            m_ih: 0.0,
            ca_a: 0.0001,
            g_na: 300.0,
            g_k: 40.0,
            g_l_s: 0.03,
            g_ca_t: 2.0,
            g_ih: 0.02,
            g_l_t: 0.03,
            g_ca_a: 1.5,
            g_kca: 2.5,
            g_l_a: 0.03,
            g_st: 1.5,
            g_ta: 0.8,
            p_s: 0.15,
            p_t: 0.25,
            p_a: 0.60,
            e_na: 50.0,
            e_k: -85.0,
            e_ca: 140.0,
            e_ih: -45.0,
            e_l: -75.0,
            ca_decay: 200.0,
            f_ca: 0.0002,
            c_m: 1.0,
            dt: 0.025,
            v_threshold: -30.0,
        }
    }
    fn valid(&self) -> bool {
        self.v_s.is_finite()
            && self.h_na.is_finite()
            && self.n_k.is_finite()
            && self.v_t.is_finite()
            && self.m_ca.is_finite()
            && self.h_ca.is_finite()
            && self.m_ih.is_finite()
            && self.v_a.is_finite()
            && self.ca_a.is_finite()
            && self.ca_a >= 0.0
            && self.g_na.is_finite()
            && self.g_na >= 0.0
            && self.g_k.is_finite()
            && self.g_k >= 0.0
            && self.g_l_s.is_finite()
            && self.g_l_s >= 0.0
            && self.g_ca_t.is_finite()
            && self.g_ca_t >= 0.0
            && self.g_ih.is_finite()
            && self.g_ih >= 0.0
            && self.g_l_t.is_finite()
            && self.g_l_t >= 0.0
            && self.g_ca_a.is_finite()
            && self.g_ca_a >= 0.0
            && self.g_kca.is_finite()
            && self.g_kca >= 0.0
            && self.g_l_a.is_finite()
            && self.g_l_a >= 0.0
            && self.g_st.is_finite()
            && self.g_st >= 0.0
            && self.g_ta.is_finite()
            && self.g_ta >= 0.0
            && self.p_s.is_finite()
            && self.p_s > 0.0
            && self.p_t.is_finite()
            && self.p_t > 0.0
            && self.p_a.is_finite()
            && self.p_a > 0.0
            && self.e_na.is_finite()
            && self.e_k.is_finite()
            && self.e_ca.is_finite()
            && self.e_ih.is_finite()
            && self.e_l.is_finite()
            && self.ca_decay.is_finite()
            && self.ca_decay > 0.0
            && self.f_ca.is_finite()
            && self.f_ca >= 0.0
            && self.c_m.is_finite()
            && self.c_m > 0.0
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.v_threshold.is_finite()
    }

    fn derivatives(&self, s: [f64; 9], current_soma: f64, current_tuft: f64) -> [f64; 9] {
        let v_s = s[0];
        let h_na = s[1];
        let n_k = s[2];
        let v_t = s[3];
        let m_ca = s[4];
        let h_ca = s[5];
        let m_ih = s[6];
        let v_a = s[7];
        let ca_a = s[8].max(0.0);

        let m_na_inf = 1.0 / (1.0 + (-(v_s + 38.0) / 7.0).exp());
        let h_na_inf = 1.0 / (1.0 + ((v_s + 65.0) / 6.0).exp());
        let n_k_inf = 1.0 / (1.0 + (-(v_s + 25.0) / 12.0).exp());
        let tau_h = 0.5 + 14.0 / (1.0 + ((v_s + 35.0) / 10.0).exp());
        let tau_n = 1.0 + 5.0 / (1.0 + ((v_s + 30.0) / 10.0).exp());
        let i_na = self.g_na * m_na_inf * m_na_inf * m_na_inf * h_na * (v_s - self.e_na);
        let i_k = self.g_k * n_k * n_k * n_k * n_k * (v_s - self.e_k);
        let i_l_s = self.g_l_s * (v_s - self.e_l);
        let i_st = self.g_st * (v_s - v_t) / self.p_s;

        let m_ca_inf = 1.0 / (1.0 + (-(v_t + 27.0) / 7.0).exp());
        let h_ca_inf = 1.0 / (1.0 + ((v_t + 52.0) / 5.0).exp());
        let m_ih_inf = 1.0 / (1.0 + ((v_t + 75.0) / 5.5).exp());
        let i_ca_t = self.g_ca_t * m_ca * m_ca * h_ca * (v_t - self.e_ca);
        let i_ih = self.g_ih * m_ih * (v_t - self.e_ih);
        let i_l_t = self.g_l_t * (v_t - self.e_l);
        let i_ts = self.g_st * (v_t - v_s) / self.p_t;
        let i_ta = self.g_ta * (v_t - v_a) / self.p_t;

        let m_ca_a_inf = 1.0 / (1.0 + (-(v_a + 30.0) / 5.0).exp());
        let kca_act = ca_a / (ca_a + 0.001);
        let i_ca_a = self.g_ca_a * m_ca_a_inf * m_ca_a_inf * (v_a - self.e_ca);
        let i_kca = self.g_kca * kca_act * (v_a - self.e_k);
        let i_l_a = self.g_l_a * (v_a - self.e_l);
        let i_at = self.g_ta * (v_a - v_t) / self.p_a;

        [
            (-i_na - i_k - i_l_s - i_st + current_soma / self.p_s) / self.c_m,
            (h_na_inf - h_na) / tau_h,
            (n_k_inf - n_k) / tau_n,
            (-i_ca_t - i_ih - i_l_t - i_ts - i_ta) / self.c_m,
            m_ca_inf - m_ca,
            (h_ca_inf - h_ca) / 20.0,
            (m_ih_inf - m_ih) / 50.0,
            (-i_ca_a - i_kca - i_l_a - i_at + current_tuft / self.p_a) / self.c_m,
            -self.f_ca * i_ca_a - ca_a / self.ca_decay,
        ]
    }

    fn rk4_substep(&self, s: [f64; 9], current_soma: f64, current_tuft: f64) -> [f64; 9] {
        let dt = self.dt;
        let k1 = self.derivatives(s, current_soma, current_tuft);
        let mut s2 = [0.0; 9];
        let mut s3 = [0.0; 9];
        let mut s4 = [0.0; 9];
        for i in 0..9 {
            s2[i] = s[i] + 0.5 * dt * k1[i];
        }
        let k2 = self.derivatives(s2, current_soma, current_tuft);
        for i in 0..9 {
            s3[i] = s[i] + 0.5 * dt * k2[i];
        }
        let k3 = self.derivatives(s3, current_soma, current_tuft);
        for i in 0..9 {
            s4[i] = s[i] + dt * k3[i];
        }
        let k4 = self.derivatives(s4, current_soma, current_tuft);
        let mut next = [0.0; 9];
        for i in 0..9 {
            next[i] = s[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }
        next[8] = next[8].max(0.0);
        next
    }

    pub fn step(&mut self, current_soma: f64, current_tuft: f64) -> i32 {
        if !current_soma.is_finite() || !current_tuft.is_finite() || !self.valid() {
            return 0;
        }
        let v_s_prev = self.v_s;
        let mut state = [
            self.v_s, self.h_na, self.n_k, self.v_t, self.m_ca, self.h_ca, self.m_ih, self.v_a,
            self.ca_a,
        ];
        for _ in 0..4 {
            state = self.rk4_substep(state, current_soma, current_tuft);
            if !state.iter().all(|value| value.is_finite()) {
                return 0;
            }
        }
        self.v_s = state[0];
        self.h_na = state[1];
        self.n_k = state[2];
        self.v_t = state[3];
        self.m_ca = state[4];
        self.h_ca = state[5];
        self.m_ih = state[6];
        self.v_a = state[7];
        self.ca_a = state[8];
        if self.v_s >= self.v_threshold && v_s_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v_s = -75.0;
        self.v_t = -75.0;
        self.v_a = -75.0;
        self.h_na = 0.9;
        self.n_k = 0.1;
        self.m_ca = 0.0;
        self.h_ca = 1.0;
        self.m_ih = 0.0;
        self.ca_a = 0.0001;
    }
}
impl Default for HayL5PyramidalNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hay_fires() {
        let mut n = HayL5PyramidalNeuron::new();
        let t: i32 = (0..500).map(|_| n.step(20.0, 0.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn hay_reset() {
        let mut n = HayL5PyramidalNeuron::new();
        for _ in 0..100 {
            n.step(20.0, 0.0);
        }
        n.reset();
        assert!((n.v_s - (-75.0)).abs() < 1e-10);
    }

    #[test]
    fn hay_bounded() {
        let mut n = HayL5PyramidalNeuron::new();
        for _ in 0..500 {
            n.step(100.0, 0.0);
        }
        assert!(n.v_s.is_finite());
    }

    #[test]
    fn hay_nan_no_panic() {
        HayL5PyramidalNeuron::new().step(f64::NAN, 0.0);
    }

    #[test]
    fn hay_rk4_somatic_anchor() {
        let mut n = HayL5PyramidalNeuron::new();
        let spikes: i32 = (0..20_000).map(|_| n.step(10.0, 0.0)).sum();
        assert_eq!(spikes, 1);
        assert!(n.ca_a >= 0.0);
    }

    #[test]
    fn hay_rk4_dual_input_anchor() {
        let mut n = HayL5PyramidalNeuron::new();
        let spikes: i32 = (0..20_000).map(|_| n.step(5.0, 5.0)).sum();
        assert_eq!(spikes, 4);
    }

    #[test]
    fn hay_invalid_input_preserves_state() {
        let mut n = HayL5PyramidalNeuron::new();
        for _ in 0..10 {
            n.step(10.0, 0.0);
        }
        let old = [
            n.v_s, n.h_na, n.n_k, n.v_t, n.m_ca, n.h_ca, n.m_ih, n.v_a, n.ca_a,
        ];
        assert_eq!(n.step(f64::INFINITY, 0.0), 0);
        assert_eq!(
            [n.v_s, n.h_na, n.n_k, n.v_t, n.m_ca, n.h_ca, n.m_ih, n.v_a, n.ca_a],
            old
        );
    }
}
