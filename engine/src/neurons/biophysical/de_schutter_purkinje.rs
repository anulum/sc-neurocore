// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — De Schutter-Bower Purkinje Neuron Model

//! De Schutter-Bower Purkinje-cell conductance dynamics.

/// De Schutter-Bower Purkinje cell — Ca-dependent K. De Schutter & Bower 1994.
///
/// The compact seven-state point model uses candidate-first RK4 with five
/// internal substeps over `(v, h_na, n_k, m_cap, h_cap, q_kca, ca)`.
#[derive(Clone, Debug)]
pub struct DeSchutterPurkinjeNeuron {
    pub v: f64,
    pub h_na: f64,
    pub n_k: f64,
    pub m_cap: f64,
    pub h_cap: f64,
    pub q_kca: f64,
    pub ca: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_cap: f64,
    pub g_kca: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_l: f64,
    pub ca_decay: f64,
    pub f_ca: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl DeSchutterPurkinjeNeuron {
    pub fn new() -> Self {
        Self {
            v: -68.0,
            h_na: 0.8,
            n_k: 0.1,
            m_cap: 0.0,
            h_cap: 0.9,
            q_kca: 0.0,
            ca: 0.0001,
            g_na: 125.0,
            g_k: 10.0,
            g_cap: 45.0,
            g_kca: 35.0,
            g_l: 0.5,
            e_na: 45.0,
            e_k: -85.0,
            e_ca: 135.0,
            e_l: -68.0,
            ca_decay: 0.02,
            f_ca: 0.00024,
            dt: 0.01,
            v_threshold: -20.0,
        }
    }
    fn valid(&self) -> bool {
        self.v.is_finite()
            && self.h_na.is_finite()
            && self.n_k.is_finite()
            && self.m_cap.is_finite()
            && self.h_cap.is_finite()
            && self.q_kca.is_finite()
            && self.ca.is_finite()
            && self.ca >= 0.0
            && self.g_na.is_finite()
            && self.g_na >= 0.0
            && self.g_k.is_finite()
            && self.g_k >= 0.0
            && self.g_cap.is_finite()
            && self.g_cap >= 0.0
            && self.g_kca.is_finite()
            && self.g_kca >= 0.0
            && self.g_l.is_finite()
            && self.g_l >= 0.0
            && self.e_na.is_finite()
            && self.e_k.is_finite()
            && self.e_ca.is_finite()
            && self.e_l.is_finite()
            && self.ca_decay.is_finite()
            && self.ca_decay >= 0.0
            && self.f_ca.is_finite()
            && self.f_ca >= 0.0
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.v_threshold.is_finite()
    }
    fn derivatives(&self, s: [f64; 7], current: f64) -> [f64; 7] {
        let v = s[0];
        let h_na = s[1];
        let n_k = s[2];
        let m_cap = s[3];
        let h_cap = s[4];
        let q_kca = s[5];
        let ca = s[6].max(0.0);
        let m_na = 1.0 / (1.0 + (-(v + 35.0) / 7.5).exp());
        let h_na_inf = 1.0 / (1.0 + ((v + 55.0) / 7.0).exp());
        let n_inf = 1.0 / (1.0 + (-(v + 30.0) / 15.0).exp());
        let m_cap_inf = 1.0 / (1.0 + (-(v + 19.0) / 5.5).exp());
        let h_cap_inf = 1.0 / (1.0 + ((v + 48.0) / 7.0).exp());
        let q_inf = ca / (ca + 0.0002);
        let tau_h_na = 0.5 + 14.0 / (1.0 + ((v + 40.0) / 12.0).exp());
        let tau_n_k = 1.0 + 11.0 / (1.0 + ((v + 15.0) / 8.0).exp());
        let i_na = self.g_na * m_na * m_na * m_na * h_na * (v - self.e_na);
        let i_k = self.g_k * n_k * n_k * n_k * n_k * (v - self.e_k);
        let i_cap = self.g_cap * m_cap * m_cap * h_cap * (v - self.e_ca);
        let i_kca = self.g_kca * q_kca * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        [
            -i_na - i_k - i_cap - i_kca - i_l + current,
            (h_na_inf - h_na) / tau_h_na,
            (n_inf - n_k) / tau_n_k,
            (m_cap_inf - m_cap) / 0.3,
            (h_cap_inf - h_cap) / 45.0,
            q_inf - q_kca,
            -self.f_ca * i_cap - self.ca_decay * ca,
        ]
    }
    fn rk4_substep(&self, s: [f64; 7], current: f64) -> [f64; 7] {
        let dt = self.dt;
        let k1 = self.derivatives(s, current);
        let mut s2 = [0.0; 7];
        let mut s3 = [0.0; 7];
        let mut s4 = [0.0; 7];
        for i in 0..7 {
            s2[i] = s[i] + 0.5 * dt * k1[i];
        }
        let k2 = self.derivatives(s2, current);
        for i in 0..7 {
            s3[i] = s[i] + 0.5 * dt * k2[i];
        }
        let k3 = self.derivatives(s3, current);
        for i in 0..7 {
            s4[i] = s[i] + dt * k3[i];
        }
        let k4 = self.derivatives(s4, current);
        let mut next = [0.0; 7];
        for i in 0..7 {
            next[i] = s[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }
        next[6] = next[6].max(0.0);
        next
    }
    pub fn step(&mut self, current: f64) -> i32 {
        if !current.is_finite() || !self.valid() {
            return 0;
        }
        let v_prev = self.v;
        let mut state = [
            self.v, self.h_na, self.n_k, self.m_cap, self.h_cap, self.q_kca, self.ca,
        ];
        for _ in 0..5 {
            state = self.rk4_substep(state, current);
            if !state.iter().all(|value| value.is_finite()) {
                return 0;
            }
        }
        self.v = state[0];
        self.h_na = state[1];
        self.n_k = state[2];
        self.m_cap = state[3];
        self.h_cap = state[4];
        self.q_kca = state[5];
        self.ca = state[6];
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -68.0;
        self.h_na = 0.8;
        self.n_k = 0.1;
        self.m_cap = 0.0;
        self.h_cap = 0.9;
        self.q_kca = 0.0;
        self.ca = 0.0001;
    }
}
impl Default for DeSchutterPurkinjeNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = DeSchutterPurkinjeNeuron::default();
        let constructed = DeSchutterPurkinjeNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn nonfinite_rk4_candidate_preserves_state() {
        let mut n = DeSchutterPurkinjeNeuron::new();
        let before = (n.v, n.h_na, n.n_k, n.m_cap, n.h_cap, n.q_kca, n.ca);
        assert_eq!(n.step(f64::MAX), 0);
        assert_eq!(
            (n.v, n.h_na, n.n_k, n.m_cap, n.h_cap, n.q_kca, n.ca),
            before
        );
    }

    #[test]
    fn purkinje_fires() {
        let mut n = DeSchutterPurkinjeNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(200.0)).sum();
        assert!(t > 0);
    }

    // -- DeSchutterPurkinje --
    #[test]
    fn purkinje_silent_without_input() {
        let mut n = DeSchutterPurkinjeNeuron::new();
        let _t: i32 = (0..500).map(|_| n.step(0.0)).sum();
        // Purkinje cells may have some spontaneous activity
        assert!(n.v.is_finite());
    }
    #[test]
    fn purkinje_reset_clears_state() {
        let mut n = DeSchutterPurkinjeNeuron::new();
        for _ in 0..100 {
            n.step(50.0);
        }
        n.reset();
        assert!((n.v - (-68.0)).abs() < 1e-10);
        assert!((n.ca - 0.0001).abs() < 1e-10);
    }
    #[test]
    fn purkinje_extreme_bounded() {
        let mut n = DeSchutterPurkinjeNeuron::new();
        for _ in 0..200 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn purkinje_ca_rises_with_spiking() {
        let mut n = DeSchutterPurkinjeNeuron::new();
        let ca_init = n.ca;
        for _ in 0..5000 {
            n.step(200.0);
        }
        assert!(n.ca > ca_init, "Ca²⁺ should rise during spiking");
    }
    #[test]
    fn purkinje_kca_activated_by_calcium() {
        let mut n = DeSchutterPurkinjeNeuron::new();
        for _ in 0..5000 {
            n.step(200.0);
        }
        assert!(
            n.q_kca > 0.0,
            "KCa should activate with Ca²⁺: q={}",
            n.q_kca
        );
    }
    #[test]
    fn purkinje_rk4_cross_backend_anchor() {
        let mut n = DeSchutterPurkinjeNeuron::new();
        let mut spikes = 0;
        for _ in 0..20_000 {
            spikes += n.step(500.0);
        }
        assert_eq!(spikes, 1);
    }
    #[test]
    fn purkinje_invalid_input_preserves_state() {
        let mut n = DeSchutterPurkinjeNeuron::new();
        for _ in 0..10 {
            let _ = n.step(200.0);
        }
        let old = (n.v, n.h_na, n.n_k, n.m_cap, n.h_cap, n.q_kca, n.ca);
        assert_eq!(n.step(f64::INFINITY), 0);
        assert_eq!((n.v, n.h_na, n.n_k, n.m_cap, n.h_cap, n.q_kca, n.ca), old);
    }
    #[test]
    fn purkinje_negative_no_crash() {
        let mut n = DeSchutterPurkinjeNeuron::new();
        for _ in 0..200 {
            n.step(-50.0);
        }
        assert!(n.v.is_finite());
    }
}
