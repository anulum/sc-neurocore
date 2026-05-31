// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for larter_breakspear

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct LarterBreakspearNeuron {
    pub v: f64,
    pub w: f64,
    pub z: f64,
    pub g_ca: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub v_ca: f64,
    pub v_na: f64,
    pub v_k: f64,
    pub v_l: f64,
    pub g_l: f64,
    pub phi: f64,
    pub tau_k: f64,
    pub b: f64,
    pub a_ee: f64,
    pub v0: f64,
    pub i_ext: f64,
    pub dt: f64,
}

impl LarterBreakspearNeuron {
    pub fn new() -> Self {
        Self {
            v: -0.5_f64,
            w: 0.0_f64,
            z: 0.0_f64,
            g_ca: 1.1_f64,
            g_na: 6.7_f64,
            g_k: 2.0_f64,
            v_ca: 1.0_f64,
            v_na: 0.53_f64,
            v_k: -0.7_f64,
            v_l: -0.5_f64,
            g_l: 0.5_f64,
            phi: 0.7_f64,
            tau_k: 1.0_f64,
            b: 0.1_f64,
            a_ee: 0.36_f64,
            v0: 0.0_f64,
            i_ext: 0.3_f64,
            dt: 0.01_f64,
        }
    }

    pub fn _m_ca(&self, v: f64) -> f64 {
        0.5 * (1.0 + ((v - (-0.01_f64)) / 0.15).tanh())
    }

    pub fn _m_na(&self, v: f64) -> f64 {
        0.5 * (1.0 + ((v - 0.12_f64) / 0.15).tanh())
    }

    pub fn _m_k(&self, v: f64) -> f64 {
        0.5 * (1.0 + ((v - self.v0) / 0.3).tanh())
    }

    fn derivatives(&self, v: f64, w: f64, z: f64, coupling: f64) -> (f64, f64, f64) {
        let i_ca = self.g_ca * self._m_ca(v) * (v - self.v_ca);
        let i_na = self.g_na * self._m_na(v) * (v - self.v_na);
        let i_k = self.g_k * w * (v - self.v_k);
        let i_l = self.g_l * (v - self.v_l);
        let dv = -i_ca - i_na - i_k - i_l + self.i_ext + coupling + self.a_ee * v;
        let dw = self.phi * (self._m_k(v) - w) / self.tau_k;
        let dz = self.b * (v + 0.5 - z);
        (dv, dw, dz)
    }

    pub fn step(&mut self, coupling: f64) -> f64 {
        if !validate_larter_breakspear(self) || !coupling.is_finite() {
            return f64::NAN;
        }

        let (v0, w0, z0) = (self.v, self.w, self.z);
        let dt = self.dt;
        let k1 = self.derivatives(v0, w0, z0, coupling);
        let k2 = self.derivatives(
            v0 + 0.5 * dt * k1.0,
            w0 + 0.5 * dt * k1.1,
            z0 + 0.5 * dt * k1.2,
            coupling,
        );
        let k3 = self.derivatives(
            v0 + 0.5 * dt * k2.0,
            w0 + 0.5 * dt * k2.1,
            z0 + 0.5 * dt * k2.2,
            coupling,
        );
        let k4 = self.derivatives(v0 + dt * k3.0, w0 + dt * k3.1, z0 + dt * k3.2, coupling);

        let mut next = self.clone();
        next.v = v0 + (dt / 6.0) * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0);
        next.w = w0 + (dt / 6.0) * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1);
        next.z = z0 + (dt / 6.0) * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2);
        if !validate_larter_breakspear(&next) {
            return f64::NAN;
        }
        *self = next;
        self.v
    }

    pub fn reset(&mut self) {
        // self.v, self.w, self.z = -0.5, 0.0, 0.0
        self.v = -0.5_f64;
        self.w = 0.0_f64;
        self.z = 0.0_f64;
        self.g_ca = 1.1_f64;
        self.g_na = 6.7_f64;
        self.g_k = 2.0_f64;
        self.v_ca = 1.0_f64;
        self.v_na = 0.53_f64;
        self.v_k = -0.7_f64;
        self.v_l = -0.5_f64;
        self.g_l = 0.5_f64;
        self.phi = 0.7_f64;
        self.tau_k = 1.0_f64;
        self.b = 0.1_f64;
        self.a_ee = 0.36_f64;
        self.v0 = 0.0_f64;
        self.i_ext = 0.3_f64;
        self.dt = 0.01_f64;
    }
}

pub fn validate_larter_breakspear(state: &LarterBreakspearNeuron) -> bool {
    state.v.is_finite()
        && state.w.is_finite()
        && state.z.is_finite()
        && state.g_ca.is_finite()
        && state.g_na.is_finite()
        && state.g_k.is_finite()
        && state.v_ca.is_finite()
        && state.v_na.is_finite()
        && state.v_k.is_finite()
        && state.v_l.is_finite()
        && state.g_l.is_finite()
        && state.phi.is_finite()
        && state.tau_k.is_finite()
        && state.b.is_finite()
        && state.a_ee.is_finite()
        && state.v0.is_finite()
        && state.i_ext.is_finite()
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.tau_k > 0.0
        && state.phi > 0.0
        && state.b > 0.0
        && state.g_ca > 0.0
        && state.g_na > 0.0
        && state.g_k > 0.0
        && state.g_l > 0.0
        && (0.0..=1.0).contains(&state.w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_larter_breakspear_new() {
        let state = LarterBreakspearNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_larter_breakspear(&state));
    }

    #[test]
    fn test_larter_breakspear_step() {
        let mut state = LarterBreakspearNeuron::new();
        let initial = state.v;
        let voltage = state.step(0.15);
        assert!(voltage.is_finite());
        assert_ne!(state.v, initial);
    }

    #[test]
    fn test_larter_breakspear_rejects_invalid_dt() {
        let mut state = LarterBreakspearNeuron::new();
        state.dt = 0.0;
        let before = state.clone();
        assert!(state.step(0.0).is_nan());
        assert_eq!(state.v, before.v);
        assert_eq!(state.w, before.w);
        assert_eq!(state.z, before.z);
    }
}
