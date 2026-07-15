// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Connor-Stevens Neuron Model

//! Connor-Stevens A-type potassium-current neuron dynamics.

/// Connor-Stevens — A-type K current for delay tuning. Connor et al. 1977.
#[derive(Clone, Debug)]
pub struct ConnorStevensNeuron {
    pub v: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
    pub a: f64,
    pub b: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_a: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_a: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl ConnorStevensNeuron {
    pub fn new() -> Self {
        Self {
            v: -68.0,
            m: 0.01,
            h: 0.99,
            n: 0.1,
            a: 0.5,
            b: 0.1,
            g_na: 120.0,
            g_k: 20.0,
            g_a: 47.7,
            g_l: 0.3,
            e_na: 55.0,
            e_k: -72.0,
            e_a: -75.0,
            e_l: -17.0,
            c_m: 1.0,
            dt: 0.01,
            v_threshold: 0.0,
        }
    }
    fn cs_safe_exp(x: f64) -> Option<f64> {
        if x.is_finite() && x <= 700.0 {
            Some(x.exp())
        } else {
            None
        }
    }

    fn cs_safe_rate(scale: f64, shift: f64, v: f64, denom: f64) -> Option<f64> {
        let delta = v + shift;
        let x = delta / denom;
        if x.abs() < 1e-9 {
            return Some(scale * denom);
        }
        let e = Self::cs_safe_exp(-x)?;
        let value = scale * delta / (1.0 - e);
        value.is_finite().then_some(value)
    }

    fn cs_valid_state(v: f64, m: f64, h: f64, n: f64, a: f64, b: f64) -> bool {
        [v, m, h, n, a, b].iter().all(|x| x.is_finite())
            && (-250.0..=250.0).contains(&v)
            && (-0.05..=1.05).contains(&m)
            && (-0.05..=1.05).contains(&h)
            && (-0.05..=1.05).contains(&n)
            && (-0.05..=1.5).contains(&a)
            && (-0.05..=1.05).contains(&b)
    }

    fn cs_valid_static(&self) -> bool {
        [
            self.g_na,
            self.g_k,
            self.g_a,
            self.g_l,
            self.e_na,
            self.e_k,
            self.e_a,
            self.e_l,
            self.c_m,
            self.dt,
            self.v_threshold,
        ]
        .iter()
        .all(|x| x.is_finite())
            && self.g_na >= 0.0
            && self.g_k >= 0.0
            && self.g_a >= 0.0
            && self.g_l >= 0.0
            && self.c_m > 0.0
            && self.dt > 0.0
    }

    fn cs_derivatives(
        &self,
        state: (f64, f64, f64, f64, f64, f64),
        current: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64)> {
        let (v, m, h, n, a, b) = state;
        let am = Self::cs_safe_rate(0.38, 29.7, v, 10.0)?;
        let bm = 15.2 * Self::cs_safe_exp(-(v + 54.7) / 18.0)?;
        let ah = 0.266 * Self::cs_safe_exp(-(v + 48.0) / 20.0)?;
        let bh = 3.8 / (1.0 + Self::cs_safe_exp(-(v + 18.0) / 10.0)?);
        let an = Self::cs_safe_rate(0.02, 45.7, v, 10.0)?;
        let bn = 0.25 * Self::cs_safe_exp(-(v + 55.7) / 80.0)?;
        let a_base = 0.0761 * Self::cs_safe_exp((v + 94.22) / 31.84)?
            / (1.0 + Self::cs_safe_exp((v + 1.17) / 28.93)?);
        if !a_base.is_finite() || a_base < 0.0 {
            return None;
        }
        let a_inf = a_base.powf(1.0 / 3.0);
        let tau_a = 0.3632 + 1.158 / (1.0 + Self::cs_safe_exp((v + 55.96) / 20.12)?);
        let b_base = 1.0 / (1.0 + Self::cs_safe_exp((v + 53.3) / 14.54)?);
        let b_inf = b_base.powf(4.0);
        let tau_b = 1.24 + 2.678 / (1.0 + Self::cs_safe_exp((v + 50.0) / 16.027)?);
        let i_na = self.g_na * m.powi(3) * h * (v - self.e_na);
        let i_k = self.g_k * n.powi(4) * (v - self.e_k);
        let i_a = self.g_a * a.powi(3) * b * (v - self.e_a);
        let i_l = self.g_l * (v - self.e_l);
        let deriv = (
            (-i_na - i_k - i_a - i_l + current) / self.c_m,
            am * (1.0 - m) - bm * m,
            ah * (1.0 - h) - bh * h,
            an * (1.0 - n) - bn * n,
            (a_inf - a) / tau_a,
            (b_inf - b) / tau_b,
        );
        [deriv.0, deriv.1, deriv.2, deriv.3, deriv.4, deriv.5]
            .iter()
            .all(|x| x.is_finite())
            .then_some(deriv)
    }

    fn cs_rk4_candidate(&self, current: f64) -> Option<(f64, f64, f64, f64, f64, f64)> {
        if !self.cs_valid_static()
            || !current.is_finite()
            || !Self::cs_valid_state(self.v, self.m, self.h, self.n, self.a, self.b)
        {
            return None;
        }
        let mut state = (self.v, self.m, self.h, self.n, self.a, self.b);
        let substeps = (1.0 / self.dt.max(0.001)) as usize;
        for _ in 0..substeps {
            let k1 = self.cs_derivatives(state, current)?;
            let k2_state = (
                state.0 + 0.5 * self.dt * k1.0,
                state.1 + 0.5 * self.dt * k1.1,
                state.2 + 0.5 * self.dt * k1.2,
                state.3 + 0.5 * self.dt * k1.3,
                state.4 + 0.5 * self.dt * k1.4,
                state.5 + 0.5 * self.dt * k1.5,
            );
            let k2 = self.cs_derivatives(k2_state, current)?;
            let k3_state = (
                state.0 + 0.5 * self.dt * k2.0,
                state.1 + 0.5 * self.dt * k2.1,
                state.2 + 0.5 * self.dt * k2.2,
                state.3 + 0.5 * self.dt * k2.3,
                state.4 + 0.5 * self.dt * k2.4,
                state.5 + 0.5 * self.dt * k2.5,
            );
            let k3 = self.cs_derivatives(k3_state, current)?;
            let k4_state = (
                state.0 + self.dt * k3.0,
                state.1 + self.dt * k3.1,
                state.2 + self.dt * k3.2,
                state.3 + self.dt * k3.3,
                state.4 + self.dt * k3.4,
                state.5 + self.dt * k3.5,
            );
            let k4 = self.cs_derivatives(k4_state, current)?;
            state = (
                state.0 + self.dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0,
                state.1 + self.dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0,
                state.2 + self.dt * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2) / 6.0,
                state.3 + self.dt * (k1.3 + 2.0 * k2.3 + 2.0 * k3.3 + k4.3) / 6.0,
                state.4 + self.dt * (k1.4 + 2.0 * k2.4 + 2.0 * k3.4 + k4.4) / 6.0,
                state.5 + self.dt * (k1.5 + 2.0 * k2.5 + 2.0 * k3.5 + k4.5) / 6.0,
            );
            if !Self::cs_valid_state(state.0, state.1, state.2, state.3, state.4, state.5) {
                return None;
            }
        }
        Some(state)
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let Some((v, m, h, n, a, b)) = self.cs_rk4_candidate(current) else {
            return 0;
        };
        self.v = v;
        self.m = m;
        self.h = h;
        self.n = n;
        self.a = a;
        self.b = b;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -68.0;
        self.m = 0.01;
        self.h = 0.99;
        self.n = 0.1;
        self.a = 0.5;
        self.b = 0.1;
    }
}
impl Default for ConnorStevensNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = ConnorStevensNeuron::default();
        let constructed = ConnorStevensNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn safe_helpers_cover_limits_and_reject_overflow() {
        assert_eq!(
            ConnorStevensNeuron::cs_safe_rate(0.38, 29.7, -29.7, 10.0),
            Some(3.8)
        );
        assert_eq!(ConnorStevensNeuron::cs_safe_exp(701.0), None);
    }

    #[test]
    fn cs_fires() {
        let mut n = ConnorStevensNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(10.0)).sum();
        assert!(t > 0);
    }

    // -- ConnorStevens --
    #[test]
    fn cs_silent_without_input() {
        let mut n = ConnorStevensNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn cs_reset_clears_state() {
        let mut n = ConnorStevensNeuron::new();
        for _ in 0..100 {
            n.step(10.0);
        }
        n.reset();
        assert!((n.v - (-68.0)).abs() < 1e-10);
    }
    #[test]
    fn cs_extreme_bounded() {
        let mut n = ConnorStevensNeuron::new();
        for _ in 0..200 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn cs_a_type_delays_spike() {
        // Connor-Stevens 1977: A-type K current causes onset delay
        // With 100 sub-steps/call, use more calls and verify A-type suppresses early firing
        let mut n_with_a = ConnorStevensNeuron::new();
        let mut n_no_a = ConnorStevensNeuron::new();
        n_no_a.g_a = 0.0;
        let spikes_with: i32 = (0..50).map(|_| n_with_a.step(8.0)).sum();
        let spikes_no: i32 = (0..50).map(|_| n_no_a.step(8.0)).sum();
        // Without A-current, neuron should fire more (no transient suppression)
        assert!(
            spikes_no >= spikes_with,
            "without A-type K, should fire more: no_a={} vs with_a={}",
            spikes_no,
            spikes_with
        );
    }
    #[test]
    fn cs_gates_bounded() {
        let mut n = ConnorStevensNeuron::new();
        for _ in 0..500 {
            n.step(10.0);
        }
        assert!(n.a >= 0.0 && n.a <= 1.5, "a={}", n.a); // a can slightly exceed 1 due to kinetics
        assert!(n.b >= 0.0 && n.b <= 1.0, "b={}", n.b);
    }
    #[test]
    fn cs_negative_no_crash() {
        let mut n = ConnorStevensNeuron::new();
        for _ in 0..200 {
            n.step(-20.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn cs_invalid_current_preserves_state() {
        let mut n = ConnorStevensNeuron::new();
        let before = (n.v, n.m, n.h, n.n, n.a, n.b);
        assert_eq!(n.step(f64::NAN), 0);
        assert_eq!((n.v, n.m, n.h, n.n, n.a, n.b), before);
    }
    #[test]
    fn cs_corrupt_runtime_state_preserves_state() {
        let mut n = ConnorStevensNeuron::new();
        n.b = f64::INFINITY;
        let before = (n.v, n.m, n.h, n.n, n.a, n.b);
        assert_eq!(n.step(6.0), 0);
        assert_eq!((n.v, n.m, n.h, n.n, n.a, n.b), before);
    }
}
