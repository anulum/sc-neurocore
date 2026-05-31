// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for renshaw_cell

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct RenshawCell {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub adapt: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_adapt: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub tau_adapt: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl RenshawCell {
    pub fn new() -> Self {
        Self {
            v: -65.0_f64,
            h: 0.8_f64,
            n: 0.1_f64,
            adapt: 0.0_f64,
            g_na: 35.0_f64,
            g_k: 9.0_f64,
            g_adapt: 5.0_f64,
            g_l: 0.12_f64,
            e_na: 55.0_f64,
            e_k: -90.0_f64,
            e_l: -65.0_f64,
            c_m: 1.0_f64,
            phi: 5.0_f64,
            tau_adapt: 50.0_f64,
            dt: 0.01_f64,
            v_threshold: -20.0_f64,
        }
    }

    fn voltage_valid(value: f64) -> bool {
        value.is_finite() && (-150.0..=100.0).contains(&value)
    }

    fn probability(value: f64) -> bool {
        value.is_finite() && (0.0..=1.0).contains(&value)
    }

    fn safe_rate(a: f64, vhalf: f64, v: f64, k: f64, fallback: f64) -> f64 {
        let d = v + vhalf;
        if d.abs() < 1e-7 {
            fallback
        } else {
            a * d / (1.0 - (-d / k).exp())
        }
    }

    fn exact_gate(previous: f64, alpha: f64, beta: f64, phi: f64, dt: f64) -> Option<f64> {
        let total = phi * (alpha + beta);
        if !previous.is_finite()
            || !alpha.is_finite()
            || !beta.is_finite()
            || !total.is_finite()
            || !dt.is_finite()
            || total <= 0.0
        {
            return None;
        }
        let steady = alpha / (alpha + beta);
        Some((steady + (previous - steady) * (-total * dt).exp()).clamp(0.0, 1.0))
    }

    fn exact_relax(previous: f64, steady: f64, tau: f64, dt: f64) -> Option<f64> {
        if !previous.is_finite()
            || !steady.is_finite()
            || !tau.is_finite()
            || !dt.is_finite()
            || tau <= 0.0
        {
            return None;
        }
        Some((steady + (previous - steady) * (-dt / tau).exp()).clamp(0.0, 1.0))
    }

    fn valid_state(&self) -> bool {
        Self::voltage_valid(self.v)
            && Self::probability(self.h)
            && Self::probability(self.n)
            && Self::probability(self.adapt)
            && self.g_na.is_finite()
            && self.g_k.is_finite()
            && self.g_adapt.is_finite()
            && self.g_l.is_finite()
            && self.e_na.is_finite()
            && self.e_k.is_finite()
            && self.e_l.is_finite()
            && self.c_m.is_finite()
            && self.phi.is_finite()
            && self.tau_adapt.is_finite()
            && self.dt.is_finite()
            && self.v_threshold.is_finite()
            && self.g_na >= 0.0
            && self.g_k >= 0.0
            && self.g_adapt >= 0.0
            && self.g_l >= 0.0
            && self.c_m > 0.0
            && self.phi > 0.0
            && self.tau_adapt > 0.0
            && self.dt > 0.0
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !i_ext.is_finite() || !self.valid_state() {
            return 0;
        }

        let v_prev = self.v;
        let mut v = self.v;
        let mut h = self.h;
        let mut n = self.n;
        let mut adapt = self.adapt;
        let n_sub = ((0.5 / self.dt.max(0.001)).max(1.0)) as usize;
        for _ in 0..n_sub {
            let am = Self::safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let bm = 4.0 * (-(v + 60.0) / 18.0).exp();
            let m_inf = am / (am + bm);
            let ah = 0.07 * (-(v + 58.0) / 20.0).exp();
            let bh = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());
            let an = Self::safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let bn = 0.125 * (-(v + 44.0) / 80.0).exp();

            let Some(h_next) = Self::exact_gate(h, ah, bh, self.phi, self.dt) else {
                return 0;
            };
            let Some(n_next) = Self::exact_gate(n, an, bn, self.phi, self.dt) else {
                return 0;
            };
            let adapt_inf = 1.0 / (1.0 + (-(v + 30.0) / 5.0).exp());
            let Some(adapt_next) = Self::exact_relax(adapt, adapt_inf, self.tau_adapt, self.dt)
            else {
                return 0;
            };

            let g_na = self.g_na * m_inf.powi(3) * h_next;
            let g_k = self.g_k * n_next.powi(4);
            let g_adapt = self.g_adapt * adapt_next;
            let g_total = g_na + g_k + g_adapt + self.g_l;
            if !g_total.is_finite() || g_total <= 0.0 {
                return 0;
            }
            let steady_v = (i_ext
                + g_na * self.e_na
                + g_k * self.e_k
                + g_adapt * self.e_k
                + self.g_l * self.e_l)
                / g_total;
            let v_next = steady_v + (v - steady_v) * (-(g_total / self.c_m) * self.dt).exp();
            if !Self::voltage_valid(v_next)
                || !Self::probability(h_next)
                || !Self::probability(n_next)
                || !Self::probability(adapt_next)
            {
                return 0;
            }

            v = v_next;
            h = h_next;
            n = n_next;
            adapt = adapt_next;
        }

        self.v = v;
        self.h = h;
        self.n = n;
        self.adapt = adapt;
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

pub fn validate_renshaw_cell(state: &RenshawCell) -> bool {
    state.valid_state()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renshaw_cell_new() {
        let state = RenshawCell::new();
        assert!(state.v.is_finite());
        assert!(validate_renshaw_cell(&state));
    }

    #[test]
    fn test_renshaw_cell_step() {
        let mut state = RenshawCell::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    fn test_gate(previous: f64, alpha: f64, beta: f64, phi: f64, dt: f64) -> f64 {
        let total = phi * (alpha + beta);
        let steady = alpha / (alpha + beta);
        (steady + (previous - steady) * (-total * dt).exp()).clamp(0.0, 1.0)
    }

    fn test_adapt(previous: f64, steady: f64, tau: f64, dt: f64) -> f64 {
        (steady + (previous - steady) * (-dt / tau).exp()).clamp(0.0, 1.0)
    }

    fn reference_step(mut n: RenshawCell, i_ext: f64) -> RenshawCell {
        let n_sub = ((0.5 / n.dt.max(0.001)).max(1.0)) as usize;
        for _ in 0..n_sub {
            let am = RenshawCell::safe_rate(0.1, 35.0, n.v, 10.0, 1.0);
            let bm = 4.0 * (-(n.v + 60.0) / 18.0).exp();
            let ah = 0.07 * (-(n.v + 58.0) / 20.0).exp();
            let bh = 1.0 / (1.0 + (-(n.v + 28.0) / 10.0).exp());
            let an = RenshawCell::safe_rate(0.01, 34.0, n.v, 10.0, 0.1);
            let bn = 0.125 * (-(n.v + 44.0) / 80.0).exp();
            let m_inf = am / (am + bm);

            n.h = test_gate(n.h, ah, bh, n.phi, n.dt);
            n.n = test_gate(n.n, an, bn, n.phi, n.dt);
            let adapt_inf = 1.0 / (1.0 + (-(n.v + 30.0) / 5.0).exp());
            n.adapt = test_adapt(n.adapt, adapt_inf, n.tau_adapt, n.dt);

            let g_na = n.g_na * m_inf.powi(3) * n.h;
            let g_k = n.g_k * n.n.powi(4);
            let g_adapt = n.g_adapt * n.adapt;
            let g_total = g_na + g_k + g_adapt + n.g_l;
            let steady_v =
                (i_ext + g_na * n.e_na + g_k * n.e_k + g_adapt * n.e_k + n.g_l * n.e_l) / g_total;
            n.v = steady_v + (n.v - steady_v) * (-(g_total / n.c_m) * n.dt).exp();
        }
        n
    }

    fn snapshot(n: &RenshawCell) -> (f64, f64, f64, f64) {
        (n.v, n.h, n.n, n.adapt)
    }

    #[test]
    fn test_renshaw_cell_exact_gate_and_conductance_step() {
        let mut state = RenshawCell::new();
        let expected = reference_step(RenshawCell::new(), 4.0);

        assert_eq!(state.step(4.0), 0);

        assert!((state.v - expected.v).abs() <= 1e-12);
        assert!((state.h - expected.h).abs() <= 1e-12);
        assert!((state.n - expected.n).abs() <= 1e-12);
        assert!((state.adapt - expected.adapt).abs() <= 1e-12);
    }

    #[test]
    fn test_renshaw_cell_rejects_invalid_current_without_mutation() {
        let mut state = RenshawCell::new();
        for _ in 0..20 {
            state.step(4.0);
        }
        let before = snapshot(&state);

        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!(snapshot(&state), before);
        assert_eq!(state.step(f64::INFINITY), 0);
        assert_eq!(snapshot(&state), before);
    }

    #[test]
    fn test_renshaw_cell_rejects_excess_current_without_mutation() {
        let mut state = RenshawCell::new();
        let before = snapshot(&state);

        assert_eq!(state.step(1.0e8), 0);

        assert_eq!(snapshot(&state), before);
    }

    #[test]
    fn test_renshaw_cell_adaptation_bounded_and_active() {
        let mut state = RenshawCell::new();
        let baseline = state.adapt;
        let spikes: i32 = (0..3000).map(|_| state.step(4.0)).sum();

        assert!(spikes > 0);
        assert!(state.adapt > baseline + 0.01);
        assert!(validate_renshaw_cell(&state));
    }
}
