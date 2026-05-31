// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for connor_stevens

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, Copy)]
struct Deriv {
    v: f64,
    m: f64,
    h: f64,
    n: f64,
    a: f64,
    b: f64,
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

    fn safe_exp(x: f64) -> Option<f64> {
        if x.is_finite() && x <= 700.0 {
            Some(x.exp())
        } else {
            None
        }
    }

    fn safe_rate(scale: f64, shift: f64, v: f64, denom: f64) -> Option<f64> {
        let delta = v + shift;
        let x = delta / denom;
        if x.abs() < 1e-9 {
            return Some(scale * denom);
        }
        let e = Self::safe_exp(-x)?;
        let value = scale * delta / (1.0 - e);
        value.is_finite().then_some(value)
    }

    fn valid_state(v: f64, m: f64, h: f64, n: f64, a: f64, b: f64) -> bool {
        [v, m, h, n, a, b].iter().all(|x| x.is_finite())
            && (-250.0..=250.0).contains(&v)
            && (-0.05..=1.05).contains(&m)
            && (-0.05..=1.05).contains(&h)
            && (-0.05..=1.05).contains(&n)
            && (-0.05..=1.5).contains(&a)
            && (-0.05..=1.05).contains(&b)
    }

    fn valid_static(&self) -> bool {
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

    fn derivatives(&self, state: Deriv, current: f64) -> Option<Deriv> {
        let am = Self::safe_rate(0.38, 29.7, state.v, 10.0)?;
        let bm = 15.2 * Self::safe_exp(-(state.v + 54.7) / 18.0)?;
        let ah = 0.266 * Self::safe_exp(-(state.v + 48.0) / 20.0)?;
        let bh = 3.8 / (1.0 + Self::safe_exp(-(state.v + 18.0) / 10.0)?);
        let an = Self::safe_rate(0.02, 45.7, state.v, 10.0)?;
        let bn = 0.25 * Self::safe_exp(-(state.v + 55.7) / 80.0)?;
        let a_base = 0.0761 * Self::safe_exp((state.v + 94.22) / 31.84)?
            / (1.0 + Self::safe_exp((state.v + 1.17) / 28.93)?);
        if !a_base.is_finite() || a_base < 0.0 {
            return None;
        }
        let a_inf = a_base.powf(1.0 / 3.0);
        let tau_a = 0.3632 + 1.158 / (1.0 + Self::safe_exp((state.v + 55.96) / 20.12)?);
        let b_base = 1.0 / (1.0 + Self::safe_exp((state.v + 53.3) / 14.54)?);
        let b_inf = b_base.powf(4.0);
        let tau_b = 1.24 + 2.678 / (1.0 + Self::safe_exp((state.v + 50.0) / 16.027)?);
        let i_na = self.g_na * state.m.powi(3) * state.h * (state.v - self.e_na);
        let i_k = self.g_k * state.n.powi(4) * (state.v - self.e_k);
        let i_a = self.g_a * state.a.powi(3) * state.b * (state.v - self.e_a);
        let i_l = self.g_l * (state.v - self.e_l);
        let deriv = Deriv {
            v: (-i_na - i_k - i_a - i_l + current) / self.c_m,
            m: am * (1.0 - state.m) - bm * state.m,
            h: ah * (1.0 - state.h) - bh * state.h,
            n: an * (1.0 - state.n) - bn * state.n,
            a: (a_inf - state.a) / tau_a,
            b: (b_inf - state.b) / tau_b,
        };
        [deriv.v, deriv.m, deriv.h, deriv.n, deriv.a, deriv.b]
            .iter()
            .all(|x| x.is_finite())
            .then_some(deriv)
    }

    fn rk4_candidate(&self, current: f64) -> Option<Deriv> {
        if !self.valid_static()
            || !current.is_finite()
            || !Self::valid_state(self.v, self.m, self.h, self.n, self.a, self.b)
        {
            return None;
        }
        let mut state = Deriv {
            v: self.v,
            m: self.m,
            h: self.h,
            n: self.n,
            a: self.a,
            b: self.b,
        };
        let substeps = (1.0 / self.dt.max(0.001)) as usize;
        for _ in 0..substeps {
            let k1 = self.derivatives(state, current)?;
            let k2 = self.derivatives(
                Deriv {
                    v: state.v + 0.5 * self.dt * k1.v,
                    m: state.m + 0.5 * self.dt * k1.m,
                    h: state.h + 0.5 * self.dt * k1.h,
                    n: state.n + 0.5 * self.dt * k1.n,
                    a: state.a + 0.5 * self.dt * k1.a,
                    b: state.b + 0.5 * self.dt * k1.b,
                },
                current,
            )?;
            let k3 = self.derivatives(
                Deriv {
                    v: state.v + 0.5 * self.dt * k2.v,
                    m: state.m + 0.5 * self.dt * k2.m,
                    h: state.h + 0.5 * self.dt * k2.h,
                    n: state.n + 0.5 * self.dt * k2.n,
                    a: state.a + 0.5 * self.dt * k2.a,
                    b: state.b + 0.5 * self.dt * k2.b,
                },
                current,
            )?;
            let k4 = self.derivatives(
                Deriv {
                    v: state.v + self.dt * k3.v,
                    m: state.m + self.dt * k3.m,
                    h: state.h + self.dt * k3.h,
                    n: state.n + self.dt * k3.n,
                    a: state.a + self.dt * k3.a,
                    b: state.b + self.dt * k3.b,
                },
                current,
            )?;
            state = Deriv {
                v: state.v + self.dt * (k1.v + 2.0 * k2.v + 2.0 * k3.v + k4.v) / 6.0,
                m: state.m + self.dt * (k1.m + 2.0 * k2.m + 2.0 * k3.m + k4.m) / 6.0,
                h: state.h + self.dt * (k1.h + 2.0 * k2.h + 2.0 * k3.h + k4.h) / 6.0,
                n: state.n + self.dt * (k1.n + 2.0 * k2.n + 2.0 * k3.n + k4.n) / 6.0,
                a: state.a + self.dt * (k1.a + 2.0 * k2.a + 2.0 * k3.a + k4.a) / 6.0,
                b: state.b + self.dt * (k1.b + 2.0 * k2.b + 2.0 * k3.b + k4.b) / 6.0,
            };
            if !Self::valid_state(state.v, state.m, state.h, state.n, state.a, state.b) {
                return None;
            }
        }
        Some(state)
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        let v_prev = self.v;
        let Some(next) = self.rk4_candidate(i_ext) else {
            return 0;
        };
        self.v = next.v;
        self.m = next.m;
        self.h = next.h;
        self.n = next.n;
        self.a = next.a;
        self.b = next.b;
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

pub fn validate_connor_stevens(state: &ConnorStevensNeuron) -> bool {
    state.valid_static()
        && ConnorStevensNeuron::valid_state(state.v, state.m, state.h, state.n, state.a, state.b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connor_stevens_new() {
        let state = ConnorStevensNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_connor_stevens(&state));
    }

    #[test]
    fn test_connor_stevens_step() {
        let mut state = ConnorStevensNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
        assert!(validate_connor_stevens(&state));
    }

    #[test]
    fn invalid_current_preserves_state() {
        let mut state = ConnorStevensNeuron::new();
        let before = state.clone();
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!(state, before);
    }

    #[test]
    fn corrupt_runtime_state_preserves_state() {
        let mut state = ConnorStevensNeuron::new();
        state.b = f64::INFINITY;
        let before = state.clone();
        assert_eq!(state.step(6.0), 0);
        assert_eq!(state, before);
    }

    #[test]
    fn reset_restores_all_gates() {
        let mut state = ConnorStevensNeuron::new();
        state.b = 0.75;
        state.reset();
        assert_eq!(state.b, 0.1);
        assert!(validate_connor_stevens(&state));
    }
}
