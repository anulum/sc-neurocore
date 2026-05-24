// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for yamada

#[derive(Debug, Clone)]
pub struct YamadaNeuron {
    pub v: f64,
    pub n: f64,
    pub q: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_q: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_q: f64,
    pub e_l: f64,
    pub tau_q: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

impl YamadaNeuron {
    pub fn new() -> Self {
        Self {
            v: -60.0_f64,
            n: 0.1_f64,
            q: 0.0_f64,
            g_na: 20.0_f64,
            g_k: 10.0_f64,
            g_q: 5.0_f64,
            g_l: 0.5_f64,
            e_na: 60.0_f64,
            e_k: -80.0_f64,
            e_q: -80.0_f64,
            e_l: -60.0_f64,
            tau_q: 300.0_f64,
            dt: 0.05_f64,
            v_threshold: -20.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !i_ext.is_finite() || !validate_yamada(self) {
            return 0;
        }

        let v_prev = self.v;
        let m_inf = sigmoid((self.v + 30.0) / 9.5);
        let n_inf = sigmoid((self.v + 30.0) / 10.0);
        let q_inf = sigmoid((self.v + 50.0) / 10.0);
        let tau_n = 1.0 + 7.5 / (1.0 + ((self.v + 40.0) / 12.0).exp());

        let i_na = self.g_na * m_inf.powi(3) * (1.0 - self.n) * (self.v - self.e_na);
        let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
        let i_q = self.g_q * self.q * (self.v - self.e_q);
        let i_l = self.g_l * (self.v - self.e_l);

        let dv = (-i_na - i_k - i_q - i_l + i_ext) * self.dt;
        let dn = (n_inf - self.n) / tau_n * self.dt;
        let dq = (q_inf - self.q) / self.tau_q * self.dt;
        let next_v = self.v + dv;
        let next_n = self.n + dn;
        let next_q = self.q + dq;
        let values = [
            m_inf, n_inf, q_inf, tau_n, i_na, i_k, i_q, i_l, dv, dn, dq, next_v, next_n, next_q,
        ];
        if values.iter().any(|value| !value.is_finite())
            || !(0.0..=1.0).contains(&next_n)
            || !(0.0..=1.0).contains(&next_q)
        {
            return 0;
        }

        self.v = next_v;
        self.n = next_n;
        self.q = next_q;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = -60.0_f64;
        self.n = 0.1_f64;
        self.q = 0.0_f64;
    }
}

pub fn validate_yamada(state: &YamadaNeuron) -> bool {
    state.v.is_finite()
        && state.n.is_finite()
        && (0.0..=1.0).contains(&state.n)
        && state.q.is_finite()
        && (0.0..=1.0).contains(&state.q)
        && state.g_na.is_finite()
        && state.g_na >= 0.0
        && state.g_k.is_finite()
        && state.g_k >= 0.0
        && state.g_q.is_finite()
        && state.g_q >= 0.0
        && state.g_l.is_finite()
        && state.g_l >= 0.0
        && state.e_na.is_finite()
        && state.e_k.is_finite()
        && state.e_q.is_finite()
        && state.e_l.is_finite()
        && state.tau_q.is_finite()
        && state.tau_q > 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.v_threshold.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yamada_new() {
        let state = YamadaNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_yamada(&state));
    }

    #[test]
    fn test_yamada_step() {
        let mut state = YamadaNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn high_current_spikes_and_preserves_gate_bounds() {
        let mut state = YamadaNeuron::new();
        let mut spikes = 0;
        for _ in 0..200_000 {
            spikes += state.step(50.0);
            assert!((0.0..=1.0).contains(&state.n));
            assert!((0.0..=1.0).contains(&state.q));
        }
        assert!(spikes >= 10);
    }

    #[test]
    fn invalid_current_does_not_mutate_state() {
        let mut state = YamadaNeuron::new();
        state.v = -55.0;
        state.n = 0.2;
        state.q = 0.1;

        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!(state.v, -55.0);
        assert_eq!(state.n, 0.2);
        assert_eq!(state.q, 0.1);
    }

    #[test]
    fn invalid_candidate_update_does_not_mutate_state() {
        let mut state = YamadaNeuron::new();
        state.v = -55.0;
        state.n = 0.2;
        state.q = 0.1;
        state.dt = 1.0e308;

        assert_eq!(state.step(1.0e308), 0);
        assert_eq!(state.v, -55.0);
        assert_eq!(state.n, 0.2);
        assert_eq!(state.q, 0.1);
    }

    #[test]
    fn reset_preserves_parameters() {
        let mut state = YamadaNeuron {
            v: -20.0,
            n: 0.8,
            q: 0.7,
            g_na: 21.0,
            g_k: 11.0,
            g_q: 6.0,
            g_l: 0.6,
            e_na: 61.0,
            e_k: -81.0,
            e_q: -82.0,
            e_l: -59.0,
            tau_q: 301.0,
            dt: 0.04,
            v_threshold: -19.0,
        };

        state.reset();

        assert_eq!(state.v, -60.0);
        assert_eq!(state.n, 0.1);
        assert_eq!(state.q, 0.0);
        assert_eq!(state.g_na, 21.0);
        assert_eq!(state.g_k, 11.0);
        assert_eq!(state.g_q, 6.0);
        assert_eq!(state.g_l, 0.6);
        assert_eq!(state.e_na, 61.0);
        assert_eq!(state.e_k, -81.0);
        assert_eq!(state.e_q, -82.0);
        assert_eq!(state.e_l, -59.0);
        assert_eq!(state.tau_q, 301.0);
        assert_eq!(state.dt, 0.04);
        assert_eq!(state.v_threshold, -19.0);
    }
}
