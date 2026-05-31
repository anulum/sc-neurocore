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

fn tau_n(v: f64) -> f64 {
    let x = (v + 40.0) / 12.0;
    if !x.is_finite() {
        return f64::NAN;
    }
    if x > 709.0 {
        return 1.0;
    }
    1.0 + 7.5 / (1.0 + x.exp())
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
        let Some((next_v, next_n, next_q)) = self.rk4_candidate(i_ext) else {
            return 0;
        };

        self.v = next_v;
        self.n = next_n;
        self.q = next_q;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    fn derivatives(&self, v: f64, n: f64, q: f64, i_ext: f64) -> Option<(f64, f64, f64)> {
        if !v.is_finite()
            || !n.is_finite()
            || !(0.0..=1.0).contains(&n)
            || !q.is_finite()
            || !(0.0..=1.0).contains(&q)
            || !i_ext.is_finite()
        {
            return None;
        }

        let m_inf = sigmoid((v + 30.0) / 9.5);
        let n_inf = sigmoid((v + 30.0) / 10.0);
        let q_inf = sigmoid((v + 50.0) / 10.0);
        let tau_n = tau_n(v);
        let i_na = self.g_na * m_inf.powi(3) * (1.0 - n) * (v - self.e_na);
        let i_k = self.g_k * n.powi(4) * (v - self.e_k);
        let i_q = self.g_q * q * (v - self.e_q);
        let i_l = self.g_l * (v - self.e_l);
        let dv = -i_na - i_k - i_q - i_l + i_ext;
        let dn = (n_inf - n) / tau_n;
        let dq = (q_inf - q) / self.tau_q;
        let values = [m_inf, n_inf, q_inf, tau_n, i_na, i_k, i_q, i_l, dv, dn, dq];
        if values.iter().any(|value| !value.is_finite()) {
            return None;
        }
        Some((dv, dn, dq))
    }

    pub fn rk4_candidate(&self, i_ext: f64) -> Option<(f64, f64, f64)> {
        if !i_ext.is_finite() || !validate_yamada(self) {
            return None;
        }

        let (v0, n0, q0) = (self.v, self.n, self.q);
        let dt = self.dt;
        let k1 = self.derivatives(v0, n0, q0, i_ext)?;
        let k2 = self.derivatives(
            v0 + 0.5 * dt * k1.0,
            n0 + 0.5 * dt * k1.1,
            q0 + 0.5 * dt * k1.2,
            i_ext,
        )?;
        let k3 = self.derivatives(
            v0 + 0.5 * dt * k2.0,
            n0 + 0.5 * dt * k2.1,
            q0 + 0.5 * dt * k2.2,
            i_ext,
        )?;
        let k4 = self.derivatives(v0 + dt * k3.0, n0 + dt * k3.1, q0 + dt * k3.2, i_ext)?;
        let next_v = v0 + dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0;
        let next_n = n0 + dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0;
        let next_q = q0 + dt * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2) / 6.0;
        if !next_v.is_finite()
            || !next_n.is_finite()
            || !(0.0..=1.0).contains(&next_n)
            || !next_q.is_finite()
            || !(0.0..=1.0).contains(&next_q)
        {
            return None;
        }
        Some((next_v, next_n, next_q))
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

    fn rk4_reference(state: &YamadaNeuron, i_ext: f64) -> (f64, f64, f64) {
        let rhs = |v: f64, n: f64, q: f64| {
            let m_inf = sigmoid((v + 30.0) / 9.5);
            let n_inf = sigmoid((v + 30.0) / 10.0);
            let q_inf = sigmoid((v + 50.0) / 10.0);
            let tau_n = 1.0 + 7.5 / (1.0 + ((v + 40.0) / 12.0).exp());
            let i_na = state.g_na * m_inf.powi(3) * (1.0 - n) * (v - state.e_na);
            let i_k = state.g_k * n.powi(4) * (v - state.e_k);
            let i_q = state.g_q * q * (v - state.e_q);
            let i_l = state.g_l * (v - state.e_l);
            (
                -i_na - i_k - i_q - i_l + i_ext,
                (n_inf - n) / tau_n,
                (q_inf - q) / state.tau_q,
            )
        };

        let (k1v, k1n, k1q) = rhs(state.v, state.n, state.q);
        let (k2v, k2n, k2q) = rhs(
            state.v + 0.5 * state.dt * k1v,
            state.n + 0.5 * state.dt * k1n,
            state.q + 0.5 * state.dt * k1q,
        );
        let (k3v, k3n, k3q) = rhs(
            state.v + 0.5 * state.dt * k2v,
            state.n + 0.5 * state.dt * k2n,
            state.q + 0.5 * state.dt * k2q,
        );
        let (k4v, k4n, k4q) = rhs(
            state.v + state.dt * k3v,
            state.n + state.dt * k3n,
            state.q + state.dt * k3q,
        );
        (
            state.v + state.dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0,
            state.n + state.dt * (k1n + 2.0 * k2n + 2.0 * k3n + k4n) / 6.0,
            state.q + state.dt * (k1q + 2.0 * k2q + 2.0 * k3q + k4q) / 6.0,
        )
    }

    #[test]
    fn rk4_candidate_matches_reference() {
        let state = YamadaNeuron {
            v: -52.0,
            n: 0.22,
            q: 0.08,
            dt: 0.025,
            ..YamadaNeuron::new()
        };
        let expected = rk4_reference(&state, 18.0);
        let candidate = state.rk4_candidate(18.0).expect("valid RK4 candidate");

        assert!((candidate.0 - expected.0).abs() < 1.0e-12);
        assert!((candidate.1 - expected.1).abs() < 1.0e-14);
        assert!((candidate.2 - expected.2).abs() < 1.0e-14);
    }

    #[test]
    fn step_commits_rk4_candidate() {
        let mut state = YamadaNeuron {
            v: -52.0,
            n: 0.22,
            q: 0.08,
            dt: 0.025,
            ..YamadaNeuron::new()
        };
        let expected = rk4_reference(&state, 18.0);

        state.step(18.0);

        assert!((state.v - expected.0).abs() < 1.0e-12);
        assert!((state.n - expected.1).abs() < 1.0e-14);
        assert!((state.q - expected.2).abs() < 1.0e-14);
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
