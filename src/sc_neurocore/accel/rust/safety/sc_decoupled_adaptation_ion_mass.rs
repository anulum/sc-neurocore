// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Retained three-state project ion-mass recurrence

#[derive(Clone, Debug, PartialEq)]
pub struct SCDecoupledAdaptationIonMassNeuron {
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
    pub v0: f64,
    pub i_ext: f64,
    pub dt: f64,
}

impl SCDecoupledAdaptationIonMassNeuron {
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
            v0: 0.0,
            i_ext: 0.3,
            dt: 0.01,
        }
    }

    fn gate(v: f64, midpoint: f64, width: f64) -> f64 {
        0.5 * (1.0 + ((v - midpoint) / width).tanh())
    }
    fn derivatives(&self, v: f64, w: f64, z: f64, coupling: f64) -> (f64, f64, f64) {
        let m_ca = Self::gate(v, -0.01, 0.15);
        let m_na = Self::gate(v, 0.12, 0.15);
        let m_k = Self::gate(v, self.v0, 0.3);
        let dv = -self.g_ca * m_ca * (v - self.v_ca)
            - self.g_na * m_na * (v - self.v_na)
            - self.g_k * w * (v - self.v_k)
            - self.g_l * (v - self.v_l)
            + self.i_ext
            + coupling
            + self.a_ee * v;
        (
            dv,
            self.phi * (m_k - w) / self.tau_k,
            self.b * (v + 0.5 - z),
        )
    }
    fn valid(&self) -> bool {
        [
            self.v, self.w, self.z, self.g_ca, self.g_na, self.g_k, self.g_l, self.v_ca, self.v_na,
            self.v_k, self.v_l, self.phi, self.tau_k, self.b, self.a_ee, self.v0, self.i_ext,
            self.dt,
        ]
        .iter()
        .all(|x| x.is_finite())
            && (0.0..=1.0).contains(&self.w)
            && [
                self.g_ca, self.g_na, self.g_k, self.g_l, self.phi, self.tau_k, self.b, self.dt,
            ]
            .iter()
            .all(|x| *x > 0.0)
    }
    pub fn step(&mut self, coupling: f64) -> Result<f64, &'static str> {
        if !self.valid() || !coupling.is_finite() {
            return Err("invalid SC ion-mass input");
        }
        let (v, w, z, dt) = (self.v, self.w, self.z, self.dt);
        let k1 = self.derivatives(v, w, z, coupling);
        let k2 = self.derivatives(
            v + 0.5 * dt * k1.0,
            w + 0.5 * dt * k1.1,
            z + 0.5 * dt * k1.2,
            coupling,
        );
        let k3 = self.derivatives(
            v + 0.5 * dt * k2.0,
            w + 0.5 * dt * k2.1,
            z + 0.5 * dt * k2.2,
            coupling,
        );
        let k4 = self.derivatives(v + dt * k3.0, w + dt * k3.1, z + dt * k3.2, coupling);
        let mut next = self.clone();
        next.v = v + dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0;
        next.w = w + dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0;
        next.z = z + dt * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2) / 6.0;
        if !next.valid() {
            return Err("invalid SC ion-mass candidate");
        }
        *self = next;
        Ok(self.v)
    }
    pub fn reset(&mut self) {
        (self.v, self.w, self.z) = (-0.5, 0.0, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_anchor() {
        let mut n = SCDecoupledAdaptationIonMassNeuron::new();
        assert!((n.step(0.0).unwrap() + 0.498_759_341_907_830_5).abs() < 1e-15);
    }
}
