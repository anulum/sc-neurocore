// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Upper Motor Neuron

//! Upper motor-neuron corticospinal dynamics.

/// Upper motor neuron — layer 5 pyramidal cell, corticospinal projection.
///
/// Biophysics: Pospischil 2008 RS parameterisation (Na+, K+, M-current)
/// with added high-threshold Ca2+ current for dendritic Ca2+ spikes.
/// Regular-spiking with adaptation. Drives alpha/gamma motor neurons
/// via corticospinal tract.
///
/// Pospischil et al., Biol. Cybern. 99(4-5), 2008 (RS variant).
/// Larkum, Trends Neurosci. 36(3), 2013 (dendritic Ca2+ spikes).
#[derive(Clone, Debug)]
pub struct UpperMotorNeuron {
    pub v: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
    pub p: f64, // M-current (Kv7) activation
    pub s: f64, // High-threshold Ca2+ activation
    // Conductances
    pub g_na: f64,
    pub g_k: f64,
    pub g_m: f64,
    pub g_ca: f64,
    pub g_l: f64,
    // Reversal potentials
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl UpperMotorNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            m: 0.05,
            h: 0.6,
            n: 0.3,
            p: 0.0,
            s: 0.0,
            g_na: 50.0,
            g_k: 5.0,
            g_m: 0.07, // M-current for adaptation (Pospischil RS)
            g_ca: 0.3, // High-threshold dendritic Ca2+
            g_l: 0.1,
            e_na: 50.0,
            e_k: -90.0,
            e_ca: 120.0,
            e_l: -70.0,
            c_m: 1.0,
            dt: 0.025,
            v_threshold: -20.0,
        }
    }

    fn finite(values: &[f64]) -> bool {
        values.iter().all(|value| value.is_finite())
    }

    fn rate_exp(value: f64) -> f64 {
        value.clamp(-60.0, 60.0).exp()
    }

    fn gate(previous: f64, alpha: f64, beta: f64, dt: f64) -> Option<f64> {
        let total = alpha + beta;
        if total <= 0.0 || !total.is_finite() {
            return None;
        }
        let steady = alpha / total;
        let next = steady + (previous - steady) * Self::rate_exp(-total * dt);
        next.is_finite().then_some(next.clamp(0.0, 1.0))
    }

    fn gate_inf(previous: f64, steady: f64, tau: f64, dt: f64) -> Option<f64> {
        if tau <= 0.0 || !tau.is_finite() {
            return None;
        }
        let next = steady + (previous - steady) * Self::rate_exp(-dt / tau);
        next.is_finite().then_some(next.clamp(0.0, 1.0))
    }

    fn valid_configuration(&self) -> bool {
        Self::finite(&[
            self.g_na,
            self.g_k,
            self.g_m,
            self.g_ca,
            self.g_l,
            self.e_na,
            self.e_k,
            self.e_ca,
            self.e_l,
            self.c_m,
            self.dt,
            self.v_threshold,
        ]) && self.g_na >= 0.0
            && self.g_k >= 0.0
            && self.g_m >= 0.0
            && self.g_ca >= 0.0
            && self.g_l >= 0.0
            && self.c_m > 0.0
            && self.dt > 0.0
    }

    fn valid_state(&self) -> bool {
        Self::finite(&[self.v, self.m, self.h, self.n, self.p, self.s])
            && (-150.0..=100.0).contains(&self.v)
            && (0.0..=1.0).contains(&self.m)
            && (0.0..=1.0).contains(&self.h)
            && (0.0..=1.0).contains(&self.n)
            && (0.0..=1.0).contains(&self.p)
            && (0.0..=1.0).contains(&self.s)
    }

    fn step_candidate(
        &self,
        v: f64,
        mut m: f64,
        mut h: f64,
        mut n: f64,
        mut p: f64,
        mut s: f64,
        current: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64)> {
        let dv = v - (-56.2);
        let x_m = dv - 13.0;
        let alpha_m = if x_m.abs() < 1e-6 {
            0.32 * 4.0
        } else {
            -0.32 * x_m / (Self::rate_exp(-x_m / 4.0) - 1.0)
        };
        // Published Pospischil beta_m numerator is V - V_T - 40; an earlier revision
        // shared the -17 offset of alpha_h, which drove the cell into depolarisation
        // block (three spikes then a fixed point near threshold for any stimulus).
        let x_bm = dv - 40.0;
        let beta_m = if x_bm.abs() < 1e-6 {
            0.28 * 5.0
        } else {
            0.28 * x_bm / (Self::rate_exp(x_bm / 5.0) - 1.0)
        };
        let alpha_h = 0.128 * Self::rate_exp(-(dv - 17.0) / 18.0);
        let beta_h = 4.0 / (1.0 + Self::rate_exp(-(dv - 40.0) / 5.0));
        let x_n = dv - 15.0;
        let alpha_n = if x_n.abs() < 1e-6 {
            0.032 * 5.0
        } else {
            -0.032 * x_n / (Self::rate_exp(-x_n / 5.0) - 1.0)
        };
        let beta_n = 0.5 * Self::rate_exp(-(dv - 10.0) / 40.0);
        m = Self::gate(m, alpha_m, beta_m, self.dt)?;
        h = Self::gate(h, alpha_h, beta_h, self.dt)?;
        n = Self::gate(n, alpha_n, beta_n, self.dt)?;
        let p_inf = 1.0 / (1.0 + Self::rate_exp(-(v + 35.0) / 10.0));
        let tau_p =
            400.0 / (3.3 * Self::rate_exp((v + 35.0) / 20.0) + Self::rate_exp(-(v + 35.0) / 20.0));
        p = Self::gate_inf(p, p_inf, tau_p, self.dt)?;
        let s_inf = 1.0 / (1.0 + Self::rate_exp(-(v + 20.0) / 5.0));
        s = Self::gate_inf(s, s_inf, 10.0, self.dt)?;
        let g_na = self.g_na * m.powi(3) * h;
        let g_k = self.g_k * n.powi(4);
        let g_m = self.g_m * p;
        let g_ca = self.g_ca * s.powi(2);
        let g_total = g_na + g_k + g_m + g_ca + self.g_l;
        if g_total <= 0.0 || !g_total.is_finite() {
            return None;
        }
        let steady_v = (current
            + g_na * self.e_na
            + g_k * self.e_k
            + g_m * self.e_k
            + g_ca * self.e_ca
            + self.g_l * self.e_l)
            / g_total;
        let next_v = steady_v + (v - steady_v) * Self::rate_exp(-(g_total / self.c_m) * self.dt);
        Self::finite(&[next_v, m, h, n, p, s]).then_some((
            next_v.clamp(-150.0, 100.0),
            m,
            h,
            n,
            p,
            s,
        ))
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !self.valid_configuration() || !self.valid_state() || !current.is_finite() {
            return 0;
        }
        let v_prev = self.v;
        let (mut v, mut m, mut h, mut n, mut p, mut s) =
            (self.v, self.m, self.h, self.n, self.p, self.s);
        for _ in 0..4 {
            let Some(next) = self.step_candidate(v, m, h, n, p, s, current) else {
                return 0;
            };
            (v, m, h, n, p, s) = next;
        }
        (self.v, self.m, self.h, self.n, self.p, self.s) = (v, m, h, n, p, s);
        if v >= self.v_threshold && v_prev < self.v_threshold {
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
        self.s = 0.0;
    }
}

impl Default for UpperMotorNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Upper Motor Neuron — 6-dimension coverage ──────────────────

    #[test]
    fn upper_motor_fires_with_input() {
        let mut n = UpperMotorNeuron::new();
        let spikes: i32 = (0..10000).map(|_| n.step(5.0)).sum();
        assert!(spikes > 0, "upper motor must fire: got {spikes}");
    }

    #[test]
    fn upper_motor_no_fire_without_input() {
        let mut n = UpperMotorNeuron::new();
        let spikes: i32 = (0..5000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    fn upper_motor_reference_step(mut n: UpperMotorNeuron, current: f64) -> UpperMotorNeuron {
        fn gate(previous: f64, alpha: f64, beta: f64, dt: f64) -> f64 {
            let total = alpha + beta;
            let steady = alpha / total;
            (steady + (previous - steady) * (-total * dt).exp()).clamp(0.0, 1.0)
        }
        fn gate_inf(previous: f64, steady: f64, tau: f64, dt: f64) -> f64 {
            (steady + (previous - steady) * (-dt / tau).exp()).clamp(0.0, 1.0)
        }
        let vt = -56.2;
        for _ in 0..4 {
            let dv = n.v - vt;
            let x_m = dv - 13.0;
            let alpha_m = if x_m.abs() < 1e-6 {
                0.32 * 4.0
            } else {
                -0.32 * x_m / ((-x_m / 4.0).exp() - 1.0)
            };
            let x_bm = dv - 40.0;
            let beta_m = if x_bm.abs() < 1e-6 {
                0.28 * 5.0
            } else {
                0.28 * x_bm / ((x_bm / 5.0).exp() - 1.0)
            };
            let alpha_h = 0.128 * (-(dv - 17.0) / 18.0).exp();
            let beta_h = 4.0 / (1.0 + (-(dv - 40.0) / 5.0).exp());
            let x_n = dv - 15.0;
            let alpha_n = if x_n.abs() < 1e-6 {
                0.032 * 5.0
            } else {
                -0.032 * x_n / ((-x_n / 5.0).exp() - 1.0)
            };
            let beta_n = 0.5 * (-(dv - 10.0) / 40.0).exp();

            n.m = gate(n.m, alpha_m, beta_m, n.dt);
            n.h = gate(n.h, alpha_h, beta_h, n.dt);
            n.n = gate(n.n, alpha_n, beta_n, n.dt);

            let p_inf = 1.0 / (1.0 + (-(n.v + 35.0) / 10.0).exp());
            let tau_p = 400.0 / (3.3 * ((n.v + 35.0) / 20.0).exp() + (-(n.v + 35.0) / 20.0).exp());
            n.p = gate_inf(n.p, p_inf, tau_p, n.dt);

            let s_inf = 1.0 / (1.0 + (-(n.v + 20.0) / 5.0).exp());
            n.s = gate_inf(n.s, s_inf, 10.0, n.dt);

            let g_na = n.g_na * n.m.powi(3) * n.h;
            let g_k = n.g_k * n.n.powi(4);
            let g_m = n.g_m * n.p;
            let g_ca = n.g_ca * n.s.powi(2);
            let g_total = g_na + g_k + g_m + g_ca + n.g_l;
            let steady_v = (current
                + g_na * n.e_na
                + g_k * n.e_k
                + g_m * n.e_k
                + g_ca * n.e_ca
                + n.g_l * n.e_l)
                / g_total;
            n.v = steady_v + (n.v - steady_v) * (-(g_total / n.c_m) * n.dt).exp();
        }
        n
    }

    #[test]
    fn upper_motor_uses_exact_gate_and_conductance_membrane_step() {
        let mut n = UpperMotorNeuron::new();
        let expected = upper_motor_reference_step(n.clone(), 5.0);

        assert_eq!(n.step(5.0), 0);

        assert!((n.v - expected.v).abs() <= 1e-12);
        assert!((n.m - expected.m).abs() <= 1e-12);
        assert!((n.h - expected.h).abs() <= 1e-12);
        assert!((n.n - expected.n).abs() <= 1e-12);
        assert!((n.p - expected.p).abs() <= 1e-12);
        assert!((n.s - expected.s).abs() <= 1e-12);
    }

    #[test]
    fn upper_motor_corrupted_gate_is_preserved_on_step() {
        let mut n = UpperMotorNeuron::new();
        n.m = 1.5;
        let before = n.clone();

        assert_eq!(n.step(5.0), 0);

        assert_eq!(n.v, before.v);
        assert_eq!(n.m, before.m);
        assert_eq!(n.h, before.h);
        assert_eq!(n.n, before.n);
        assert_eq!(n.p, before.p);
        assert_eq!(n.s, before.s);
    }

    #[test]
    fn upper_motor_negative_current_no_fire() {
        let mut n = UpperMotorNeuron::new();
        let spikes: i32 = (0..2000).map(|_| n.step(-5.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn upper_motor_adaptation_via_m_current() {
        let mut n = UpperMotorNeuron::new();
        let first: i32 = (0..5000).map(|_| n.step(5.0)).sum();
        let second: i32 = (0..5000).map(|_| n.step(5.0)).sum();
        assert!(
            second <= first + 3,
            "M-current should cause adaptation: first={first}, second={second}"
        );
    }

    #[test]
    fn upper_motor_ca_activates_during_depolarisation() {
        let mut n = UpperMotorNeuron::new();
        let baseline = n.s;
        for _ in 0..5000 {
            n.step(5.0);
        }
        assert!(
            n.s > baseline + 0.001,
            "Ca2+ gate should activate: s={}",
            n.s
        );
    }

    #[test]
    fn upper_motor_reset_roundtrip() {
        let mut n = UpperMotorNeuron::new();
        for _ in 0..3000 {
            n.step(5.0);
        }
        n.reset();
        let mut fresh = UpperMotorNeuron::new();
        let r1: i32 = (0..2000).map(|_| n.step(5.0)).sum();
        let r2: i32 = (0..2000).map(|_| fresh.step(5.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn upper_motor_voltage_bounded() {
        let mut n = UpperMotorNeuron::new();
        for _ in 0..20000 {
            n.step(10.0);
        }
        assert!(n.v.is_finite());
        assert!(n.p.is_finite());
        assert!(n.s.is_finite());
    }

    #[test]
    fn upper_motor_nan_recovery() {
        let mut n = UpperMotorNeuron::new();
        for _ in 0..100 {
            n.step(5.0);
        }
        for _ in 0..10 {
            let _ = n.step(f64::NAN);
        }
        n.reset();
        assert!(n.v.is_finite());
    }

    #[test]
    fn upper_motor_extreme_input() {
        let mut n = UpperMotorNeuron::new();
        for _ in 0..50 {
            n.step(1e6);
        }
        n.reset();
        assert!(n.v.is_finite());
    }

    #[test]
    fn upper_motor_performance() {
        let mut n = UpperMotorNeuron::new();
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            n.step(5.0);
        }
        assert!(
            start.elapsed().as_millis() < 100,
            "10k steps took {:?}",
            start.elapsed()
        );
    }
}
