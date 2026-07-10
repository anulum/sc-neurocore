// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for hodgkin_huxley

//! Fail-closed Hodgkin-Huxley (1952) safety kernel.
//!
//! Mirrors the historical explicit-Euler path of `models/hodgkin_huxley.py`
//! (`_step_baseline_euler`): one macro step advances `round(1/dt)` explicit
//! sub-steps in which the gating variables update first and the membrane
//! voltage then uses the freshly-updated gates. Every sub-step is validated;
//! any non-finite intermediate leaves the caller's state untouched.

#[derive(Debug, Clone, PartialEq)]
pub struct HodgkinHuxleyNeuron {
    pub v: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
    pub c_m: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl HodgkinHuxleyNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            m: 0.05,
            h: 0.6,
            n: 0.32,
            c_m: 1.0,
            g_na: 120.0,
            g_k: 36.0,
            g_l: 0.3,
            e_na: 50.0,
            e_k: -77.0,
            e_l: -54.4,
            dt: 0.01,
            v_threshold: 0.0,
        }
    }

    fn safe_exp(x: f64) -> Option<f64> {
        if x.is_finite() && x <= 700.0 {
            Some(x.exp())
        } else {
            None
        }
    }

    /// Singular-limit opening rate `scale·d / (1 − e^(−d/denom))` with `d = v + shift`,
    /// returning the analytic limit `scale·denom` when `|d| < 1e-7` (bit-for-bit the
    /// `abs(d) < 1e-7` guard of `models/hodgkin_huxley.py`).
    fn opening_rate(scale: f64, shift: f64, denom: f64, limit: f64, v: f64) -> Option<f64> {
        let d = v + shift;
        if d.abs() < 1e-7 {
            return Some(limit);
        }
        let e = Self::safe_exp(-d / denom)?;
        let value = scale * d / (1.0 - e);
        value.is_finite().then_some(value)
    }

    fn alpha_m(v: f64) -> Option<f64> {
        Self::opening_rate(0.1, 40.0, 10.0, 1.0, v)
    }

    fn beta_m(v: f64) -> Option<f64> {
        Some(4.0 * Self::safe_exp(-(v + 65.0) / 18.0)?)
    }

    fn alpha_h(v: f64) -> Option<f64> {
        Some(0.07 * Self::safe_exp(-(v + 65.0) / 20.0)?)
    }

    fn beta_h(v: f64) -> Option<f64> {
        Some(1.0 / (1.0 + Self::safe_exp(-(v + 35.0) / 10.0)?))
    }

    fn alpha_n(v: f64) -> Option<f64> {
        Self::opening_rate(0.01, 55.0, 10.0, 0.1, v)
    }

    fn beta_n(v: f64) -> Option<f64> {
        Some(0.125 * Self::safe_exp(-(v + 65.0) / 80.0)?)
    }

    fn valid_state(v: f64, m: f64, h: f64, n: f64) -> bool {
        [v, m, h, n].iter().all(|x| x.is_finite())
            && (-250.0..=250.0).contains(&v)
            && (-0.05..=1.05).contains(&m)
            && (-0.05..=1.05).contains(&h)
            && (-0.05..=1.05).contains(&n)
    }

    fn valid_static(&self) -> bool {
        [
            self.c_m,
            self.g_na,
            self.g_k,
            self.g_l,
            self.e_na,
            self.e_k,
            self.e_l,
            self.dt,
            self.v_threshold,
        ]
        .iter()
        .all(|x| x.is_finite())
            && self.g_na >= 0.0
            && self.g_k >= 0.0
            && self.g_l >= 0.0
            && self.c_m > 0.0
            && self.dt > 0.0
    }

    /// One macro step of the explicit-Euler sub-step schedule, computed into a
    /// candidate that is only committed by `step`. Fail-closed: returns `None`
    /// (state untouched) on any non-finite parameter, current, or intermediate.
    fn euler_candidate(&self, current: f64) -> Option<(f64, f64, f64, f64)> {
        if !self.valid_static()
            || !current.is_finite()
            || !Self::valid_state(self.v, self.m, self.h, self.n)
        {
            return None;
        }
        let (mut v, mut m, mut h, mut n) = (self.v, self.m, self.h, self.n);
        let substeps = (1.0 / self.dt).round() as usize;
        for _ in 0..substeps {
            let am = Self::alpha_m(v)?;
            let bm = Self::beta_m(v)?;
            let ah = Self::alpha_h(v)?;
            let bh = Self::beta_h(v)?;
            let an = Self::alpha_n(v)?;
            let bn = Self::beta_n(v)?;
            m += (am * (1.0 - m) - bm * m) * self.dt;
            h += (ah * (1.0 - h) - bh * h) * self.dt;
            n += (an * (1.0 - n) - bn * n) * self.dt;
            let i_na = self.g_na * m.powi(3) * h * (v - self.e_na);
            let i_k = self.g_k * n.powi(4) * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);
            v += (-i_na - i_k - i_l + current) / self.c_m * self.dt;
            if !Self::valid_state(v, m, h, n) {
                return None;
            }
        }
        Some((v, m, h, n))
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        let v_prev = self.v;
        let Some((v, m, h, n)) = self.euler_candidate(i_ext) else {
            return 0;
        };
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
        self.v = -65.0;
        self.m = 0.05;
        self.h = 0.6;
        self.n = 0.32;
    }
}

impl Default for HodgkinHuxleyNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_hodgkin_huxley(state: &HodgkinHuxleyNeuron) -> bool {
    state.valid_static() && HodgkinHuxleyNeuron::valid_state(state.v, state.m, state.h, state.n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hodgkin_huxley_new() {
        let state = HodgkinHuxleyNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_hodgkin_huxley(&state));
    }

    #[test]
    fn test_hodgkin_huxley_step() {
        let mut state = HodgkinHuxleyNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
        assert!(validate_hodgkin_huxley(&state));
    }

    #[test]
    fn invalid_current_preserves_state() {
        let mut state = HodgkinHuxleyNeuron::new();
        let before = state.clone();
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!(state, before);
    }

    #[test]
    fn corrupt_runtime_state_preserves_state() {
        let mut state = HodgkinHuxleyNeuron::new();
        state.h = f64::INFINITY;
        let before = state.clone();
        assert_eq!(state.step(6.0), 0);
        assert_eq!(state, before);
    }

    #[test]
    fn reset_restores_gates() {
        let mut state = HodgkinHuxleyNeuron::new();
        state.step(20.0);
        state.reset();
        assert_eq!(state.v, -65.0);
        assert_eq!(state.m, 0.05);
        assert_eq!(state.h, 0.6);
        assert_eq!(state.n, 0.32);
        assert!(validate_hodgkin_huxley(&state));
    }

    #[test]
    fn matches_python_golden_spike_count() {
        // Parity with models/hodgkin_huxley.py (default baseline_euler integrator, 100 explicit-
        // Euler sub-steps per macro step): silent at zero drive, six action potentials at I=10 over
        // 100 macro steps, and nine at I=20. Hodgkin-Huxley gating is exp-based, so the trace is not
        // bit-exact across libms; the spike count is the stable observable and is the parity
        // contract — not a "spike is 0 or 1" smoke check. The Go and Julia kernels reproduce the
        // same counts; the Rust engine RK4 binding is covered separately by the Python parity suite.
        for (current, want) in [(0.0_f64, 0_usize), (10.0, 6), (20.0, 9)] {
            let mut state = HodgkinHuxleyNeuron::new();
            let spikes = (0..100).filter(|_| state.step(current) == 1).count();
            assert_eq!(spikes, want, "I={current}");
        }
    }
}
