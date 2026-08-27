// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety Larter-Breakspear source dynamics

/// Complete state and configuration for the Larter-Breakspear cortical mass.
#[derive(Debug, Clone, PartialEq)]
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
    pub t_ca: f64,
    pub t_na: f64,
    pub t_k: f64,
    pub delta_ca: f64,
    pub delta_na: f64,
    pub delta_k: f64,
    pub phi: f64,
    pub tau_k: f64,
    pub b: f64,
    pub a_ee: f64,
    pub a_ei: f64,
    pub a_ie: f64,
    pub a_ne: f64,
    pub a_ni: f64,
    pub r_nmda: f64,
    pub coupling_balance: f64,
    pub v_t: f64,
    pub z_t: f64,
    pub delta_v: f64,
    pub delta_z: f64,
    pub q_v_max: f64,
    pub q_z_max: f64,
    pub i_ext: f64,
    pub t_scale: f64,
    pub dt: f64,
}

impl LarterBreakspearNeuron {
    /// Construct the maintained source-profile initial state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            v: 0.1,
            w: 0.1,
            z: 0.1,
            g_ca: 1.1,
            g_na: 6.7,
            g_k: 2.0,
            g_l: 0.5,
            v_ca: 1.0,
            v_na: 0.53,
            v_k: -0.7,
            v_l: -0.5,
            t_ca: -0.01,
            t_na: 0.3,
            t_k: 0.0,
            delta_ca: 0.15,
            delta_na: 0.15,
            delta_k: 0.3,
            phi: 0.7,
            tau_k: 1.0,
            b: 0.1,
            a_ee: 0.4,
            a_ei: 2.0,
            a_ie: 2.0,
            a_ne: 1.0,
            a_ni: 0.4,
            r_nmda: 0.25,
            coupling_balance: 0.1,
            v_t: 0.0,
            z_t: 0.0,
            delta_v: 0.65,
            delta_z: 0.7,
            q_v_max: 1.0,
            q_z_max: 1.0,
            i_ext: 0.3,
            t_scale: 1.0,
            dt: 0.01,
        }
    }

    fn sigmoid(value: f64, threshold: f64, width: f64) -> f64 {
        0.5 * (1.0 + ((value - threshold) / width).tanh())
    }

    fn derivatives(&self, v: f64, w: f64, z: f64, coupling: f64) -> (f64, f64, f64) {
        let m_ca = Self::sigmoid(v, self.t_ca, self.delta_ca);
        let m_na = Self::sigmoid(v, self.t_na, self.delta_na);
        let m_k = Self::sigmoid(v, self.t_k, self.delta_k);
        let q_v = self.q_v_max * Self::sigmoid(v, self.v_t, self.delta_v);
        let q_z = self.q_z_max * Self::sigmoid(z, self.z_t, self.delta_z);
        let excitation =
            self.a_ee * ((1.0 - self.coupling_balance) * q_v + self.coupling_balance * coupling);
        let dv = -(self.g_ca + self.r_nmda * excitation) * m_ca * (v - self.v_ca)
            - self.g_k * w * (v - self.v_k)
            - self.g_l * (v - self.v_l)
            - (self.g_na * m_na + excitation) * (v - self.v_na)
            - self.a_ie * z * q_z
            + self.a_ne * self.i_ext;
        let dw = self.phi * (m_k - w) / self.tau_k;
        let dz = self.b * (self.a_ni * self.i_ext + self.a_ei * v * q_v);
        (self.t_scale * dv, self.t_scale * dw, self.t_scale * dz)
    }

    /// Advance one fixed-step classical-RK4 transition atomically.
    pub fn step(&mut self, coupling: f64) -> Result<f64, &'static str> {
        if !validate_larter_breakspear(self) || !coupling.is_finite() {
            return Err("invalid Larter-Breakspear state, configuration, or coupling");
        }
        let (v0, w0, z0, dt) = (self.v, self.w, self.z, self.dt);
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
        let candidate = (
            v0 + dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0,
            w0 + dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0,
            z0 + dt * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2) / 6.0,
        );
        let mut next = self.clone();
        (next.v, next.w, next.z) = candidate;
        if !validate_larter_breakspear(&next) {
            return Err("Larter-Breakspear candidate state is invalid");
        }
        *self = next;
        Ok(self.v)
    }

    /// Restore source-profile dynamic state while preserving configuration.
    pub fn reset(&mut self) {
        (self.v, self.w, self.z) = (0.1, 0.1, 0.1);
    }
}

impl Default for LarterBreakspearNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate all public state and configuration fields.
#[must_use]
pub fn validate_larter_breakspear(state: &LarterBreakspearNeuron) -> bool {
    let values = [
        state.v,
        state.w,
        state.z,
        state.g_ca,
        state.g_na,
        state.g_k,
        state.g_l,
        state.v_ca,
        state.v_na,
        state.v_k,
        state.v_l,
        state.t_ca,
        state.t_na,
        state.t_k,
        state.delta_ca,
        state.delta_na,
        state.delta_k,
        state.phi,
        state.tau_k,
        state.b,
        state.a_ee,
        state.a_ei,
        state.a_ie,
        state.a_ne,
        state.a_ni,
        state.r_nmda,
        state.coupling_balance,
        state.v_t,
        state.z_t,
        state.delta_v,
        state.delta_z,
        state.q_v_max,
        state.q_z_max,
        state.i_ext,
        state.t_scale,
        state.dt,
    ];
    values.iter().all(|value| value.is_finite())
        && (0.0..=1.0).contains(&state.w)
        && [
            state.g_ca,
            state.g_na,
            state.g_k,
            state.g_l,
            state.r_nmda,
            state.q_v_max,
            state.q_z_max,
        ]
        .iter()
        .all(|value| *value >= 0.0)
        && [
            state.delta_ca,
            state.delta_na,
            state.delta_k,
            state.delta_v,
            state.delta_z,
            state.phi,
            state.tau_k,
            state.b,
            state.t_scale,
            state.dt,
        ]
        .iter()
        .all(|value| *value > 0.0)
        && (0.0..=1.0).contains(&state.coupling_balance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_anchor_and_atomic_failure() {
        let mut state = LarterBreakspearNeuron::new();
        assert!((state.step(0.0).unwrap() - 0.108_510_239_033_112_84).abs() < 1.0e-15);
        let before = state.clone();
        assert!(state.step(f64::NAN).is_err());
        assert_eq!(state, before);
    }

    #[test]
    fn reset_preserves_configuration() {
        let mut state = LarterBreakspearNeuron::new();
        state.g_ca = 1.2;
        state.step(0.2).unwrap();
        state.reset();
        assert_eq!((state.v, state.w, state.z), (0.1, 0.1, 0.1));
        assert_eq!(state.g_ca, 1.2);
    }
}
