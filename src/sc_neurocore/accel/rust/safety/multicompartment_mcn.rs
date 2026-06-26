// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for multicompartment_mcn candidate-first RK4

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct MulticompartmentMCNNeuron {
    pub tau: f64,
    pub tau_b: f64,
    pub tau_a: f64,
    pub g_ratio: f64,
    pub beta: f64,
    pub v_th: f64,
    pub dt: f64,
    pub u: f64,
    pub v_basal: f64,
    pub v_apical: f64,
}

impl MulticompartmentMCNNeuron {
    pub fn new() -> Self {
        Self {
            tau: 2.0_f64,
            tau_b: 2.0_f64,
            tau_a: 2.0_f64,
            g_ratio: 1.0_f64,
            beta: 1.0_f64,
            v_th: 1.0_f64,
            dt: 1.0_f64,
            u: 0.0_f64,
            v_basal: 0.0_f64,
            v_apical: 0.0_f64,
        }
    }

    pub fn _sigma(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-self.beta * x).exp())
    }

    fn valid(&self) -> bool {
        self.tau.is_finite()
            && self.tau > 0.0
            && self.tau_b.is_finite()
            && self.tau_b > 0.0
            && self.tau_a.is_finite()
            && self.tau_a > 0.0
            && self.g_ratio.is_finite()
            && self.g_ratio >= 0.0
            && self.beta.is_finite()
            && self.beta > 0.0
            && self.v_th.is_finite()
            && self.v_th > 0.0
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.u.is_finite()
            && self.v_basal.is_finite()
            && self.v_apical.is_finite()
    }

    fn derivatives(
        &self,
        u: f64,
        v_basal: f64,
        v_apical: f64,
        x_basal: f64,
        x_apical: f64,
        i_soma: f64,
    ) -> [f64; 3] {
        let gate = self._sigma(v_apical);
        let du = (-u + gate * (self.g_ratio * (v_basal - u) + i_soma)) / self.tau;
        let dv_basal = (-v_basal + x_basal) / self.tau_b;
        let dv_apical = (-v_apical + x_apical) / self.tau_a;
        [du, dv_basal, dv_apical]
    }

    fn rk4_substep(&self, s: [f64; 3], x_basal: f64, x_apical: f64, i_soma: f64) -> [f64; 3] {
        let dt = self.dt;
        let k1 = self.derivatives(s[0], s[1], s[2], x_basal, x_apical, i_soma);
        let k2 = self.derivatives(
            s[0] + 0.5 * dt * k1[0],
            s[1] + 0.5 * dt * k1[1],
            s[2] + 0.5 * dt * k1[2],
            x_basal,
            x_apical,
            i_soma,
        );
        let k3 = self.derivatives(
            s[0] + 0.5 * dt * k2[0],
            s[1] + 0.5 * dt * k2[1],
            s[2] + 0.5 * dt * k2[2],
            x_basal,
            x_apical,
            i_soma,
        );
        let k4 = self.derivatives(
            s[0] + dt * k3[0],
            s[1] + dt * k3[1],
            s[2] + dt * k3[2],
            x_basal,
            x_apical,
            i_soma,
        );
        [
            s[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            s[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            s[2] + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
        ]
    }

    pub fn step_compartments(&mut self, x_basal: f64, x_apical: f64, i_soma: f64) -> i32 {
        if !x_basal.is_finite() || !x_apical.is_finite() || !i_soma.is_finite() || !self.valid() {
            return 0;
        }
        let next = self.rk4_substep([self.u, self.v_basal, self.v_apical], x_basal, x_apical, i_soma);
        if !next.iter().all(|value| value.is_finite()) {
            return 0;
        }
        let spike = next[0] >= self.v_th;
        self.u = if spike { 0.0 } else { next[0] };
        self.v_basal = next[1];
        self.v_apical = next[2];
        i32::from(spike)
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        self.step_compartments(i_ext, 0.0, 0.0)
    }

    pub fn reset(&mut self) {
        self.u = 0.0_f64;
        self.v_basal = 0.0_f64;
        self.v_apical = 0.0_f64;
    }
}

pub fn validate_multicompartment_mcn(state: &MulticompartmentMCNNeuron) -> bool {
    state.valid()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multicompartment_mcn_new() {
        let state = MulticompartmentMCNNeuron::new();
        assert!(validate_multicompartment_mcn(&state));
    }

    #[test]
    fn test_multicompartment_mcn_step() {
        let mut state = MulticompartmentMCNNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_multicompartment_mcn_sigma_gate() {
        let state = MulticompartmentMCNNeuron::new();
        assert!((state._sigma(0.0) - 0.5).abs() < 1.0e-12);
        assert!(state._sigma(10.0) > 0.99);
        assert!(state._sigma(-10.0) < 0.01);
    }

    #[test]
    fn test_multicompartment_mcn_cross_backend_anchor() {
        let mut state = MulticompartmentMCNNeuron::new();
        let mut spikes = 0_i32;
        for _ in 0..200_000 {
            spikes += state.step(3.2);
        }
        assert_eq!(spikes, 49_999);
    }

    #[test]
    fn test_multicompartment_mcn_invalid_input_preserves_state() {
        let mut state = MulticompartmentMCNNeuron::new();
        for _ in 0..5 {
            let _ = state.step(3.0);
        }
        let old = (state.u, state.v_basal, state.v_apical);
        assert_eq!(state.step(f64::INFINITY), 0);
        assert_eq!((state.u, state.v_basal, state.v_apical), old);
    }
}
