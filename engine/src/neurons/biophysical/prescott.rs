// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Prescott Neuron Model

//! Prescott conductance model for Type I, II, and III excitability.

/// Prescott 2008 — Type I/II/III excitability tuning via M-current.
#[derive(Clone, Debug)]
pub struct PrescottNeuron {
    pub v: f64,
    pub w: f64,
    pub g_fast: f64,
    pub g_slow: f64,
    pub g_l: f64,
    pub e_fast: f64,
    pub e_slow: f64,
    pub e_l: f64,
    pub beta_w: f64,
    pub gamma_w: f64,
    pub tau_w: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl PrescottNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            w: 0.0,
            g_fast: 20.0,
            g_slow: 20.0,
            g_l: 2.0,
            e_fast: 50.0,
            e_slow: -100.0,
            e_l: -70.0,
            beta_w: -21.0,
            gamma_w: 15.0,
            tau_w: 100.0,
            phi: 0.15,
            dt: 0.1,
            v_threshold: -20.0,
        }
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

    fn valid_state(v: f64, w: f64) -> bool {
        v.is_finite() && w.is_finite() && (0.0..=1.0).contains(&w)
    }

    fn valid_runtime(&self) -> bool {
        Self::valid_state(self.v, self.w)
            && self.g_fast.is_finite()
            && self.g_fast >= 0.0
            && self.g_slow.is_finite()
            && self.g_slow >= 0.0
            && self.g_l.is_finite()
            && self.g_l >= 0.0
            && self.e_fast.is_finite()
            && self.e_slow.is_finite()
            && self.e_l.is_finite()
            && self.beta_w.is_finite()
            && self.gamma_w.is_finite()
            && self.gamma_w > 0.0
            && self.tau_w.is_finite()
            && self.tau_w > 0.0
            && self.phi.is_finite()
            && self.phi >= 0.0
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.v_threshold.is_finite()
    }

    fn derivatives(&self, v: f64, w: f64, current: f64) -> Option<(f64, f64)> {
        if !Self::valid_state(v, w) {
            return None;
        }
        let m_inf = Self::sigmoid((v + 20.0) / 15.0);
        let w_inf = Self::sigmoid((v - self.beta_w) / self.gamma_w);
        let i_fast = self.g_fast * m_inf * (v - self.e_fast);
        let i_slow = self.g_slow * w * (v - self.e_slow);
        let i_l = self.g_l * (v - self.e_l);
        let dv = -i_fast - i_slow - i_l + current;
        let dw = self.phi * (w_inf - w) / self.tau_w;
        if dv.is_finite() && dw.is_finite() {
            Some((dv, dw))
        } else {
            None
        }
    }

    fn rk4_step(&self, current: f64) -> Option<(f64, f64)> {
        let dt = self.dt;
        let (k1_v, k1_w) = self.derivatives(self.v, self.w, current)?;
        let (k2_v, k2_w) =
            self.derivatives(self.v + 0.5 * dt * k1_v, self.w + 0.5 * dt * k1_w, current)?;
        let (k3_v, k3_w) =
            self.derivatives(self.v + 0.5 * dt * k2_v, self.w + 0.5 * dt * k2_w, current)?;
        let (k4_v, k4_w) = self.derivatives(self.v + dt * k3_v, self.w + dt * k3_w, current)?;
        let next_v = self.v + dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0;
        let next_w = self.w + dt * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w) / 6.0;
        if Self::valid_state(next_v, next_w) {
            Some((next_v, next_w))
        } else {
            None
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !current.is_finite() || !self.valid_runtime() {
            return 0;
        }
        let v_prev = self.v;
        let Some((next_v, next_w)) = self.rk4_step(current) else {
            return 0;
        };
        self.v = next_v;
        self.w = next_w;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = -65.0;
        self.w = 0.0;
    }
}
impl Default for PrescottNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = PrescottNeuron::default();
        let constructed = PrescottNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn derivative_rejects_invalid_and_nonfinite_candidates() {
        let mut n = PrescottNeuron::new();
        assert_eq!(n.derivatives(n.v, 2.0, 0.0), None);
        n.g_fast = f64::MAX;
        assert_eq!(n.derivatives(f64::MAX, 0.5, 0.0), None);
    }

    #[test]
    fn invalid_rk4_candidate_preserves_state() {
        let mut n = PrescottNeuron::new();
        n.dt = 1.0e-300;
        let before = (n.v, n.w);
        assert_eq!(n.step(f64::MAX / 2.0), 0);
        assert_eq!((n.v, n.w), before);
    }

    #[test]
    fn prescott_fires() {
        let mut n = PrescottNeuron::new();
        let t: i32 = (0..500).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }

    // -- Prescott --
    #[test]
    fn prescott_zero_input_stable() {
        let mut n = PrescottNeuron::new();
        let _t: i32 = (0..500).map(|_| n.step(0.0)).sum();
        // Prescott has fast Na conductance — may produce spontaneous activity
        assert!(n.v.is_finite());
    }
    #[test]
    fn prescott_reset_clears_state() {
        let mut n = PrescottNeuron::new();
        for _ in 0..100 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.v - (-65.0)).abs() < 1e-10);
        assert!((n.w - 0.0).abs() < 1e-10);
    }
    #[test]
    fn prescott_rk4_reference_point() {
        let mut n = PrescottNeuron::new();
        assert_eq!(n.step(50.0), 0);
        assert!((n.v - (-44.498914201492525)).abs() < 1e-12);
        assert!((n.w - 1.4035864179018786e-05).abs() < 1e-17);
    }
    #[test]
    fn prescott_extreme_bounded() {
        let mut n = PrescottNeuron::new();
        for _ in 0..200 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn prescott_slow_var_adapts() {
        let mut n = PrescottNeuron::new();
        for _ in 0..500 {
            n.step(5.0);
        }
        assert!(n.w > 0.0, "slow variable w should activate during spiking");
    }
    #[test]
    fn prescott_negative_no_crash() {
        let mut n = PrescottNeuron::new();
        for _ in 0..200 {
            n.step(-10.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn prescott_nan_no_panic() {
        let mut n = PrescottNeuron::new();
        n.step(f64::NAN);
    }
}
