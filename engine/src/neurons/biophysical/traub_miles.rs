// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Traub-Miles Neuron Model

//! Traub-Miles hippocampal CA3 pyramidal neuron dynamics.

/// Traub-Miles — hippocampal CA3 pyramidal neuron with M-current.
///
/// Full Traub et al. 1991 model with Na, K_dr, leak, plus
/// M-current (Kv7/KCNQ) for spike frequency adaptation.
/// The M-current is the slow K⁺ conductance responsible for:
/// - Spike frequency adaptation (SFA)
/// - Theta-frequency resonance (~4-8 Hz)
/// - Subthreshold membrane potential oscillations
///
/// IM = g_M * w * (V - E_K)
/// dw/dt = (w_inf - w) / tau_w
/// w_inf = 1 / (1 + exp(-(V + 35) / 10))
/// tau_w = 100 / (3.3 * exp((V+35)/20) + exp(-(V+35)/20))
///
/// Traub et al., J Neurophysiol 66:635, 1991.
/// Yamada et al., Meth Neuronal Model (Koch & Segev), 1989 (M-current kinetics).
#[derive(Clone, Debug)]
pub struct TraubMilesNeuron {
    pub v: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl TraubMilesNeuron {
    pub fn new() -> Self {
        Self {
            v: -67.0,
            m: 0.05,
            h: 0.6,
            n: 0.3,
            g_na: 100.0,
            g_k: 80.0,
            g_l: 0.1,
            e_na: 50.0,
            e_k: -100.0,
            e_l: -67.0,
            dt: 0.01,
            v_threshold: -20.0,
        }
    }
    fn finite_gate(value: f64) -> bool {
        value.is_finite() && (0.0..=1.0).contains(&value)
    }
    fn valid_runtime(&self) -> bool {
        self.v.is_finite()
            && Self::finite_gate(self.m)
            && Self::finite_gate(self.h)
            && Self::finite_gate(self.n)
            && self.g_na.is_finite()
            && self.g_na >= 0.0
            && self.g_k.is_finite()
            && self.g_k >= 0.0
            && self.g_l.is_finite()
            && self.g_l >= 0.0
            && self.e_na.is_finite()
            && self.e_k.is_finite()
            && self.e_l.is_finite()
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.v_threshold.is_finite()
    }
    fn rates(v: f64) -> Option<(f64, f64, f64, f64, f64, f64)> {
        let d = v + 54.0;
        let am = if d.abs() > 1e-6 {
            0.32 * d / (1.0 - (-d / 4.0).exp())
        } else {
            8.0
        };
        let d2 = v + 27.0;
        let bm = if d2.abs() > 1e-6 {
            0.28 * d2 / ((d2 / 5.0).exp() - 1.0)
        } else {
            5.6
        };
        let ah = 0.128 * (-(v + 50.0) / 18.0).exp();
        let bh = 4.0 / (1.0 + (-(v + 27.0) / 5.0).exp());
        let d3 = v + 52.0;
        let an = if d3.abs() > 1e-6 {
            0.032 * d3 / (1.0 - (-d3 / 5.0).exp())
        } else {
            0.32
        };
        let bn = 0.5 * (-(v + 57.0) / 40.0).exp();
        if [am, bm, ah, bh, an, bn]
            .iter()
            .all(|rate| rate.is_finite() && *rate >= 0.0)
        {
            Some((am, bm, ah, bh, an, bn))
        } else {
            None
        }
    }
    fn derivatives(
        &self,
        v: f64,
        m: f64,
        h: f64,
        n: f64,
        current: f64,
    ) -> Option<(f64, f64, f64, f64)> {
        if !v.is_finite() || !Self::finite_gate(m) || !Self::finite_gate(h) || !Self::finite_gate(n)
        {
            return None;
        }
        let (am, bm, ah, bh, an, bn) = Self::rates(v)?;
        let dm = am * (1.0 - m) - bm * m;
        let dh = ah * (1.0 - h) - bh * h;
        let dn = an * (1.0 - n) - bn * n;
        let i_na = self.g_na * m.powi(3) * h * (v - self.e_na);
        let i_k = self.g_k * n.powi(4) * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        let dv = -i_na - i_k - i_l + current;
        if [dv, dm, dh, dn, i_na, i_k, i_l]
            .iter()
            .all(|value| value.is_finite())
        {
            Some((dv, dm, dh, dn))
        } else {
            None
        }
    }
    fn rk4_substep(
        &self,
        v: f64,
        m: f64,
        h: f64,
        n: f64,
        current: f64,
    ) -> Option<(f64, f64, f64, f64)> {
        let (k1_v, k1_m, k1_h, k1_n) = self.derivatives(v, m, h, n, current)?;
        let (k2_v, k2_m, k2_h, k2_n) = self.derivatives(
            v + 0.5 * self.dt * k1_v,
            m + 0.5 * self.dt * k1_m,
            h + 0.5 * self.dt * k1_h,
            n + 0.5 * self.dt * k1_n,
            current,
        )?;
        let (k3_v, k3_m, k3_h, k3_n) = self.derivatives(
            v + 0.5 * self.dt * k2_v,
            m + 0.5 * self.dt * k2_m,
            h + 0.5 * self.dt * k2_h,
            n + 0.5 * self.dt * k2_n,
            current,
        )?;
        let (k4_v, k4_m, k4_h, k4_n) = self.derivatives(
            v + self.dt * k3_v,
            m + self.dt * k3_m,
            h + self.dt * k3_h,
            n + self.dt * k3_n,
            current,
        )?;
        let next_v = v + self.dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0;
        let next_m = m + self.dt * (k1_m + 2.0 * k2_m + 2.0 * k3_m + k4_m) / 6.0;
        let next_h = h + self.dt * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h) / 6.0;
        let next_n = n + self.dt * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n) / 6.0;
        if next_v.is_finite()
            && Self::finite_gate(next_m)
            && Self::finite_gate(next_h)
            && Self::finite_gate(next_n)
        {
            Some((next_v, next_m, next_h, next_n))
        } else {
            None
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        if !current.is_finite() || !self.valid_runtime() {
            return 0;
        }
        let v_prev = self.v;
        let mut v = self.v;
        let mut m = self.m;
        let mut h = self.h;
        let mut n = self.n;
        for _ in 0..10 {
            let Some((next_v, next_m, next_h, next_n)) = self.rk4_substep(v, m, h, n, current)
            else {
                return 0;
            };
            v = next_v;
            m = next_m;
            h = next_h;
            n = next_n;
        }
        self.v = v;
        self.m = m;
        self.h = h;
        self.n = n;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}
impl Default for TraubMilesNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = TraubMilesNeuron::default();
        let constructed = TraubMilesNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn removable_rate_singularities_use_finite_limits() {
        for voltage in [-54.0, -27.0, -52.0] {
            assert!(TraubMilesNeuron::rates(voltage).is_some());
        }
        assert!(TraubMilesNeuron::rates(-1.0e308).is_none());
    }

    #[test]
    fn derivatives_reject_invalid_and_overflowing_states() {
        let mut n = TraubMilesNeuron::new();
        assert_eq!(n.derivatives(n.v, 2.0, n.h, n.n, 0.0), None);
        n.e_na = -f64::MAX;
        assert_eq!(n.derivatives(f64::MAX, 0.5, 0.5, 0.5, 0.0), None);
    }

    #[test]
    fn invalid_rk4_candidate_preserves_state() {
        for (voltage, dt, current) in [
            (-200.0, 0.0001, 1.0e100),
            (-150.0, 0.05623413251903491, 10_000.0),
            (-200.0, 0.0031622776601683794, 0.0),
            (0.0, 0.03162277660168379, 10_000.0),
        ] {
            let mut n = TraubMilesNeuron::new();
            n.v = voltage;
            n.dt = dt;
            let before = (n.v, n.m, n.h, n.n);
            assert_eq!(n.step(current), 0);
            assert_eq!((n.v, n.m, n.h, n.n), before);
        }
    }

    #[test]
    fn traub_fires() {
        let mut n = TraubMilesNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }

    // -- TraubMiles --
    #[test]
    fn traub_silent_without_input() {
        let mut n = TraubMilesNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn traub_reset_clears_state() {
        let mut n = TraubMilesNeuron::new();
        for _ in 0..100 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.v - (-67.0)).abs() < 1e-10);
    }
    #[test]
    fn traub_extreme_bounded() {
        let mut n = TraubMilesNeuron::new();
        for _ in 0..200 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn traub_gates_bounded() {
        let mut n = TraubMilesNeuron::new();
        for _ in 0..500 {
            n.step(5.0);
        }
        assert!(n.m >= 0.0 && n.m <= 1.01);
        assert!(n.h >= 0.0 && n.h <= 1.01);
        assert!(n.n >= 0.0 && n.n <= 1.01);
    }
    #[test]
    fn traub_weak_negative_no_crash() {
        let mut n = TraubMilesNeuron::new();
        for _ in 0..200 {
            n.step(-5.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn traub_nan_no_panic() {
        let mut n = TraubMilesNeuron::new();
        n.step(f64::NAN);
    }
    #[test]
    fn traub_rk4_reference_point() {
        let mut n = TraubMilesNeuron::new();
        n.v = -63.5;
        n.m = 0.08;
        n.h = 0.55;
        n.n = 0.32;
        let spike = n.step(4.0);
        assert_eq!(spike, 0);
        assert!((n.v - (-65.6638958700765)).abs() < 1e-13);
        assert!((n.m - 0.04237301812907925).abs() < 1e-15);
        assert!((n.h - 0.5626824931070477).abs() < 1e-15);
        assert!((n.n - 0.30356298261126924).abs() < 1e-15);
        assert!((n.v - (-65.66233161606698)).abs() > 1e-3);
    }
}
