// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for dendritic NMDA candidate-first RK4

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct DendriticNMDANeuron {
    pub g_nmda: f64,
    pub e_nmda: f64,
    pub mg_conc: f64,
    pub g_coupling: f64,
    pub tau_soma: f64,
    pub tau_dend: f64,
    pub theta: f64,
    pub dt: f64,
    pub v_soma: f64,
    pub v_dend: f64,
}

impl DendriticNMDANeuron {
    pub fn new() -> Self {
        Self {
            g_nmda: 1.5,
            e_nmda: 0.0,
            mg_conc: 1.0,
            g_coupling: 0.5,
            tau_soma: 20.0,
            tau_dend: 50.0,
            theta: -50.0,
            dt: 0.1,
            v_soma: -65.0,
            v_dend: -65.0,
        }
    }

    fn valid(&self) -> bool {
        self.g_nmda.is_finite()
            && self.g_nmda >= 0.0
            && self.e_nmda.is_finite()
            && self.mg_conc.is_finite()
            && self.mg_conc >= 0.0
            && self.g_coupling.is_finite()
            && self.g_coupling >= 0.0
            && self.tau_soma.is_finite()
            && self.tau_soma > 0.0
            && self.tau_dend.is_finite()
            && self.tau_dend > 0.0
            && self.theta.is_finite()
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.v_soma.is_finite()
            && self.v_dend.is_finite()
    }

    pub fn mg_block(&self, v: f64) -> f64 {
        1.0 / (1.0 + (self.mg_conc / 3.57) * (-0.062 * v).exp())
    }

    fn derivatives(&self, v_soma: f64, v_dend: f64, i_soma: f64, glutamate: f64) -> (f64, f64) {
        let block = self.mg_block(v_dend);
        let i_nmda = self.g_nmda * glutamate * block * (v_dend - self.e_nmda);
        let dv_soma =
            (-v_soma - 65.0 + i_soma + self.g_coupling * (v_dend - v_soma)) / self.tau_soma;
        let dv_dend =
            (-v_dend - 65.0 + i_nmda + self.g_coupling * (v_soma - v_dend)) / self.tau_dend;
        (dv_soma, dv_dend)
    }

    fn rk4_substep(&self, v_soma: f64, v_dend: f64, i_soma: f64, glutamate: f64) -> (f64, f64) {
        let dt = self.dt;
        let (k1s, k1d) = self.derivatives(v_soma, v_dend, i_soma, glutamate);
        let (k2s, k2d) = self.derivatives(
            v_soma + 0.5 * dt * k1s,
            v_dend + 0.5 * dt * k1d,
            i_soma,
            glutamate,
        );
        let (k3s, k3d) = self.derivatives(
            v_soma + 0.5 * dt * k2s,
            v_dend + 0.5 * dt * k2d,
            i_soma,
            glutamate,
        );
        let (k4s, k4d) = self.derivatives(v_soma + dt * k3s, v_dend + dt * k3d, i_soma, glutamate);
        (
            v_soma + dt * (k1s + 2.0 * k2s + 2.0 * k3s + k4s) / 6.0,
            v_dend + dt * (k1d + 2.0 * k2d + 2.0 * k3d + k4d) / 6.0,
        )
    }

    pub fn step(&mut self, i_soma: f64, glutamate: f64) -> i32 {
        if !i_soma.is_finite() || !glutamate.is_finite() || glutamate < 0.0 || !self.valid() {
            return 0;
        }
        let (next_v_soma, next_v_dend) =
            self.rk4_substep(self.v_soma, self.v_dend, i_soma, glutamate);
        if !next_v_soma.is_finite() || !next_v_dend.is_finite() {
            return 0;
        }
        self.v_dend = next_v_dend;
        if next_v_soma >= self.theta {
            self.v_soma = -65.0;
            1
        } else {
            self.v_soma = next_v_soma;
            0
        }
    }

    pub fn reset(&mut self) {
        self.v_soma = -65.0;
        self.v_dend = -65.0;
    }
}

pub fn validate_dendritic_nmda(state: &DendriticNMDANeuron) -> bool {
    state.valid()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dendritic_nmda_new() {
        let state = DendriticNMDANeuron::new();
        assert!(validate_dendritic_nmda(&state));
    }

    #[test]
    fn test_dendritic_nmda_step() {
        let mut state = DendriticNMDANeuron::new();
        let spike = state.step(10.0, 0.5);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_dendritic_nmda_cross_backend_anchor() {
        let mut state = DendriticNMDANeuron::new();
        let mut spikes = 0_i32;
        for _ in 0..20_000 {
            spikes += state.step(50.0, 0.5);
        }
        assert_eq!(spikes, 253);
        assert!(state.v_soma.is_finite());
        assert!(state.v_dend.is_finite());
    }

    #[test]
    fn test_dendritic_nmda_invalid_input_preserves_state() {
        let mut state = DendriticNMDANeuron::new();
        for _ in 0..10 {
            let _ = state.step(50.0, 0.5);
        }
        let old = (state.v_soma, state.v_dend);
        assert_eq!(state.step(f64::INFINITY, 0.5), 0);
        assert_eq!((state.v_soma, state.v_dend), old);
        assert_eq!(state.step(50.0, -1.0), 0);
        assert_eq!((state.v_soma, state.v_dend), old);
    }

    #[test]
    fn test_dendritic_nmda_invalid_configuration_preserves_state() {
        let mut state = DendriticNMDANeuron::new();
        for _ in 0..10 {
            let _ = state.step(50.0, 0.5);
        }
        let old = (state.v_soma, state.v_dend);
        state.tau_dend = 0.0;
        assert_eq!(state.step(50.0, 0.5), 0);
        assert_eq!((state.v_soma, state.v_dend), old);
    }
}
