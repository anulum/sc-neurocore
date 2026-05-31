// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for upper_motor_neuron

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct UpperMotorNeuron {
    pub v: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
    pub p: f64,
    pub s: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_m: f64,
    pub g_ca: f64,
    pub g_l: f64,
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
            v: -70.0_f64,
            m: 0.05_f64,
            h: 0.6_f64,
            n: 0.3_f64,
            p: 0.0_f64,
            s: 0.0_f64,
            g_na: 50.0_f64,
            g_k: 5.0_f64,
            g_m: 0.07_f64,
            g_ca: 0.3_f64,
            g_l: 0.1_f64,
            e_na: 50.0_f64,
            e_k: -90.0_f64,
            e_ca: 120.0_f64,
            e_l: -70.0_f64,
            c_m: 1.0_f64,
            dt: 0.025_f64,
            v_threshold: -20.0_f64,
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
        i_ext: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64)> {
        let dv = v - (-56.2);
        let x_m = dv - 13.0;
        let alpha_m = if x_m.abs() < 1e-6 {
            0.32 * 4.0
        } else {
            -0.32 * x_m / (Self::rate_exp(-x_m / 4.0) - 1.0)
        };
        let x_h = dv - 17.0;
        let beta_m = if x_h.abs() < 1e-6 {
            0.28 * 5.0
        } else {
            0.28 * x_h / (Self::rate_exp(x_h / 5.0) - 1.0)
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
        let steady_v = (i_ext
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

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !self.valid_configuration() || !self.valid_state() || !i_ext.is_finite() {
            return 0;
        }
        let v_prev = self.v;
        let (mut v, mut m, mut h, mut n, mut p, mut s) =
            (self.v, self.m, self.h, self.n, self.p, self.s);
        for _ in 0..4 {
            let Some(next) = self.step_candidate(v, m, h, n, p, s, i_ext) else {
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
        // self.v = -70.0
        // self.m = 0.05
        // self.h = 0.6
        // self.n = 0.3
        // self.p = 0.0
        // self.s = 0.0
        self.v = -70.0_f64;
        self.m = 0.05_f64;
        self.h = 0.6_f64;
        self.n = 0.3_f64;
        self.p = 0.0_f64;
        self.s = 0.0_f64;
    }
}

pub fn validate_upper_motor_neuron(state: &UpperMotorNeuron) -> bool {
    state.valid_configuration() && state.valid_state()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_step(mut state: UpperMotorNeuron, current: f64) -> UpperMotorNeuron {
        fn gate(previous: f64, alpha: f64, beta: f64, dt: f64) -> f64 {
            let total = alpha + beta;
            let steady = alpha / total;
            (steady + (previous - steady) * (-total * dt).exp()).clamp(0.0, 1.0)
        }
        fn gate_inf(previous: f64, steady: f64, tau: f64, dt: f64) -> f64 {
            (steady + (previous - steady) * (-dt / tau).exp()).clamp(0.0, 1.0)
        }
        let vt = -56.2_f64;
        for _ in 0..4 {
            let dv = state.v - vt;
            let x_m = dv - 13.0;
            let alpha_m = if x_m.abs() < 1e-6 {
                0.32 * 4.0
            } else {
                -0.32 * x_m / ((-x_m / 4.0).exp() - 1.0)
            };
            let x_h = dv - 17.0;
            let beta_m = if x_h.abs() < 1e-6 {
                0.28 * 5.0
            } else {
                0.28 * x_h / ((x_h / 5.0).exp() - 1.0)
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
            state.m = gate(state.m, alpha_m, beta_m, state.dt);
            state.h = gate(state.h, alpha_h, beta_h, state.dt);
            state.n = gate(state.n, alpha_n, beta_n, state.dt);
            let p_inf = 1.0 / (1.0 + (-(state.v + 35.0) / 10.0).exp());
            let tau_p =
                400.0 / (3.3 * ((state.v + 35.0) / 20.0).exp() + (-(state.v + 35.0) / 20.0).exp());
            state.p = gate_inf(state.p, p_inf, tau_p, state.dt);
            let s_inf = 1.0 / (1.0 + (-(state.v + 20.0) / 5.0).exp());
            state.s = gate_inf(state.s, s_inf, 10.0, state.dt);
            let g_na = state.g_na * state.m.powi(3) * state.h;
            let g_k = state.g_k * state.n.powi(4);
            let g_m = state.g_m * state.p;
            let g_ca = state.g_ca * state.s.powi(2);
            let g_total = g_na + g_k + g_m + g_ca + state.g_l;
            let steady_v = (current
                + g_na * state.e_na
                + g_k * state.e_k
                + g_m * state.e_k
                + g_ca * state.e_ca
                + state.g_l * state.e_l)
                / g_total;
            state.v = steady_v + (state.v - steady_v) * (-(g_total / state.c_m) * state.dt).exp();
        }
        state
    }

    #[test]
    fn test_upper_motor_neuron_new() {
        let state = UpperMotorNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_upper_motor_neuron(&state));
    }

    #[test]
    fn test_upper_motor_neuron_step() {
        let mut state = UpperMotorNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn step_matches_exact_gate_and_conductance_membrane_reference() {
        let mut state = UpperMotorNeuron::new();
        let expected = reference_step(state.clone(), 5.0);

        assert_eq!(state.step(5.0), 0);

        assert!((state.v - expected.v).abs() <= 1e-12);
        assert!((state.m - expected.m).abs() <= 1e-12);
        assert!((state.h - expected.h).abs() <= 1e-12);
        assert!((state.n - expected.n).abs() <= 1e-12);
        assert!((state.p - expected.p).abs() <= 1e-12);
        assert!((state.s - expected.s).abs() <= 1e-12);
    }

    #[test]
    fn invalid_current_preserves_state() {
        let mut state = UpperMotorNeuron::new();
        state.v = -64.0;
        state.m = 0.1;
        let before = state.clone();

        assert_eq!(state.step(f64::NAN), 0);

        assert_eq!(state.v, before.v);
        assert_eq!(state.m, before.m);
        assert_eq!(state.h, before.h);
        assert_eq!(state.n, before.n);
        assert_eq!(state.p, before.p);
        assert_eq!(state.s, before.s);
    }

    #[test]
    fn corrupted_gate_preserved_on_step() {
        let mut state = UpperMotorNeuron::new();
        state.m = 1.5;
        let before = state.clone();

        assert_eq!(state.step(5.0), 0);

        assert_eq!(state.v, before.v);
        assert_eq!(state.m, before.m);
        assert_eq!(state.h, before.h);
        assert_eq!(state.n, before.n);
        assert_eq!(state.p, before.p);
        assert_eq!(state.s, before.s);
    }
}
