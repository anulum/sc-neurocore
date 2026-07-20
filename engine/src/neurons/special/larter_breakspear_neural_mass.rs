// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Larter-Breakspear neural-mass model

//! Conductance-based Larter-Breakspear whole-brain neural-mass dynamics.

/// Larter-Breakspear 2003 — neural mass with ion channels for whole-brain modelling.
#[derive(Clone, Debug)]
pub struct LarterBreakspearNeuron {
    pub v: f64,
    pub w: f64,
    pub z: f64,
    pub g_ca: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub v_ca: f64,
    pub v_na: f64,
    pub v_k: f64,
    pub v_l: f64,
    pub phi: f64,
    pub tau_k: f64,
    pub b: f64,
    pub a_ee: f64,
    pub i_ext: f64,
    pub dt: f64,
}

impl LarterBreakspearNeuron {
    pub fn new() -> Self {
        Self {
            v: -0.5,
            w: 0.0,
            z: 0.0,
            g_ca: 1.1,
            g_na: 6.7,
            g_k: 2.0,
            g_l: 0.5,
            v_ca: 1.0,
            v_na: 0.53,
            v_k: -0.7,
            v_l: -0.5,
            phi: 0.7,
            tau_k: 1.0,
            b: 0.1,
            a_ee: 0.36,
            i_ext: 0.3,
            dt: 0.01,
        }
    }
    pub fn step(&mut self, coupling: f64) -> f64 {
        let m_ca = 0.5 * (1.0 + ((self.v + 0.01) / 0.15).tanh());
        let m_na = 0.5 * (1.0 + ((self.v - 0.12) / 0.15).tanh());
        let m_k = 0.5 * (1.0 + (self.v / 0.3).tanh());
        let i_ca = self.g_ca * m_ca * (self.v - self.v_ca);
        let i_na = self.g_na * m_na * (self.v - self.v_na);
        let i_k = self.g_k * self.w * (self.v - self.v_k);
        let i_l = self.g_l * (self.v - self.v_l);
        let dv = -i_ca - i_na - i_k - i_l + self.i_ext + coupling + self.a_ee * self.v;
        let dw = self.phi * (m_k - self.w) / self.tau_k;
        let dz = self.b * (self.v + 0.5 - self.z);
        self.v += dv * self.dt;
        self.w += dw * self.dt;
        self.z += dz * self.dt;
        self.v
    }
    pub fn reset(&mut self) {
        self.v = -0.5;
        self.w = 0.0;
        self.z = 0.0;
    }
}

impl Default for LarterBreakspearNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn membrane_state_evolves() {
        let mut n = LarterBreakspearNeuron::new();
        let initial_voltage = n.v;
        for _ in 0..1000 {
            n.step(0.0);
        }
        assert!((n.v - initial_voltage).abs() > 0.001);
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut n = LarterBreakspearNeuron::new();
        for _ in 0..500 {
            n.step(0.0);
        }
        n.reset();
        assert!((n.v - (-0.5)).abs() < 1e-10);
        assert!((n.w - 0.0).abs() < 1e-10);
        assert!((n.z - 0.0).abs() < 1e-10);
    }

    #[test]
    fn membrane_state_remains_finite() {
        let mut n = LarterBreakspearNeuron::new();
        for _ in 0..5000 {
            n.step(10.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn nan_coupling_does_not_panic() {
        LarterBreakspearNeuron::new().step(f64::NAN);
    }
}
