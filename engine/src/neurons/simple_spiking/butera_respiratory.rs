// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Butera Respiratory Neuron Model

//! Butera pre-Botzinger respiratory neuron dynamics.

/// Butera pre-Bötzinger respiratory neuron. Butera et al. 1999.
#[derive(Clone, Debug)]
pub struct ButeraRespiratoryNeuron {
    pub v: f64,
    pub n: f64,
    pub h_nap: f64,
    pub g_na: f64,
    pub g_nap: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub tau_h: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl ButeraRespiratoryNeuron {
    pub fn new() -> Self {
        Self {
            v: -50.0,
            n: 0.01,
            h_nap: 0.5,
            g_na: 28.0,
            g_nap: 2.8,
            g_k: 11.2,
            g_l: 2.8,
            e_na: 50.0,
            e_k: -85.0,
            e_l: -65.0,
            tau_h: 10000.0,
            dt: 0.1,
            v_threshold: -20.0,
        }
    }
    fn butera_valid_state(v: f64, n: f64, h_nap: f64) -> bool {
        [v, n, h_nap].iter().all(|x| x.is_finite())
            && (-200.0..=100.0).contains(&v)
            && (-0.05..=1.05).contains(&n)
            && (-0.05..=1.05).contains(&h_nap)
    }

    fn butera_valid_static(&self) -> bool {
        [
            self.g_na,
            self.g_nap,
            self.g_k,
            self.g_l,
            self.e_na,
            self.e_k,
            self.e_l,
            self.tau_h,
            self.dt,
            self.v_threshold,
        ]
        .iter()
        .all(|x| x.is_finite())
            && self.g_na >= 0.0
            && self.g_nap >= 0.0
            && self.g_k >= 0.0
            && self.g_l >= 0.0
            && self.tau_h > 0.0
            && self.dt > 0.0
    }

    fn butera_derivatives(&self, state: (f64, f64, f64), current: f64) -> Option<(f64, f64, f64)> {
        let (mut v, mut n, mut h_nap) = state;
        if !(v.is_finite() && n.is_finite() && h_nap.is_finite() && current.is_finite()) {
            return None;
        }
        v = v.clamp(-200.0, 100.0);
        n = n.clamp(0.0, 1.0);
        h_nap = h_nap.clamp(0.0, 1.0);
        let m_na = 1.0 / (1.0 + (-(v + 34.0) / 5.0).exp());
        let n_inf = 1.0 / (1.0 + (-(v + 29.0) / 4.0).exp());
        let m_nap = 1.0 / (1.0 + (-(v + 40.0) / 6.0).exp());
        let h_nap_inf = 1.0 / (1.0 + ((v + 48.0) / 6.0).exp());
        let tau_n = (10.0 / ((v + 29.0) / 8.0).cosh().max(1e-12)).max(0.01);
        let tau_h_eff = (self.tau_h / ((v + 48.0) / 12.0).cosh().max(1e-12)).max(0.1);
        let i_na = self.g_na * m_na.powi(3) * (1.0 - n) * (v - self.e_na);
        let i_nap = self.g_nap * m_nap * h_nap * (v - self.e_na);
        let i_k = self.g_k * n.powi(4) * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        let deriv = (
            -i_na - i_nap - i_k - i_l + current,
            (n_inf - n) / tau_n,
            (h_nap_inf - h_nap) / tau_h_eff,
        );
        [deriv.0, deriv.1, deriv.2]
            .iter()
            .all(|x| x.is_finite())
            .then_some(deriv)
    }

    fn butera_rk4_candidate(&self, current: f64) -> Option<(f64, f64, f64)> {
        if !self.butera_valid_static()
            || !current.is_finite()
            || !Self::butera_valid_state(self.v, self.n, self.h_nap)
        {
            return None;
        }
        let state = (self.v, self.n, self.h_nap);
        let k1 = self.butera_derivatives(state, current)?;
        let k2_state = (
            state.0 + 0.5 * self.dt * k1.0,
            state.1 + 0.5 * self.dt * k1.1,
            state.2 + 0.5 * self.dt * k1.2,
        );
        let k2 = self.butera_derivatives(k2_state, current)?;
        let k3_state = (
            state.0 + 0.5 * self.dt * k2.0,
            state.1 + 0.5 * self.dt * k2.1,
            state.2 + 0.5 * self.dt * k2.2,
        );
        let k3 = self.butera_derivatives(k3_state, current)?;
        let k4_state = (
            state.0 + self.dt * k3.0,
            state.1 + self.dt * k3.1,
            state.2 + self.dt * k3.2,
        );
        let k4 = self.butera_derivatives(k4_state, current)?;
        let candidate = (
            state.0 + self.dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0,
            state.1 + self.dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0,
            state.2 + self.dt * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2) / 6.0,
        );
        if candidate.0.is_finite() && candidate.1.is_finite() && candidate.2.is_finite() {
            Some((
                candidate.0.clamp(-200.0, 100.0),
                candidate.1.clamp(0.0, 1.0),
                candidate.2.clamp(0.0, 1.0),
            ))
        } else {
            None
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let Some((v, n, h_nap)) = self.butera_rk4_candidate(current) else {
            return 0;
        };
        self.v = v;
        self.n = n;
        self.h_nap = h_nap;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -50.0;
        self.n = 0.01;
        self.h_nap = 0.5;
    }
}
impl Default for ButeraRespiratoryNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = ButeraRespiratoryNeuron::default();
        let constructed = ButeraRespiratoryNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn butera_fires() {
        let mut n = ButeraRespiratoryNeuron::new();
        let t: i32 = (0..20000).map(|_| n.step(50.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn butera_reset_clears_state() {
        let mut n = ButeraRespiratoryNeuron::new();
        for _ in 0..1000 {
            n.step(50.0);
        }
        n.reset();
        assert!((n.v - (-50.0)).abs() < 1e-10);
    }

    #[test]
    fn butera_bounded() {
        let mut n = ButeraRespiratoryNeuron::new();
        for _ in 0..5000 {
            n.step(500.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn butera_nan_no_panic() {
        ButeraRespiratoryNeuron::new().step(f64::NAN);
    }

    #[test]
    fn butera_nan_preserves_state() {
        let mut n = ButeraRespiratoryNeuron::new();
        let before = (n.v, n.n, n.h_nap);
        assert_eq!(n.step(f64::NAN), 0);
        assert_eq!((n.v, n.n, n.h_nap), before);
    }

    #[test]
    fn butera_negative_no_crash() {
        let mut n = ButeraRespiratoryNeuron::new();
        for _ in 0..500 {
            n.step(-20.0);
        }
        assert!(n.v.is_finite());
    }
}
