// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for neurogrid candidate-first RK4

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct NeuroGridNeuron {
    pub v_s: f64,
    pub v_d: f64,
    pub tau_s: f64,
    pub tau_d: f64,
    pub g_c: f64,
    pub delta_t: f64,
    pub v_rest: f64,
    pub v_threshold: f64,
    pub v_peak: f64,
    pub v_reset: f64,
    pub dt: f64,
}

impl NeuroGridNeuron {
    pub fn new() -> Self {
        Self {
            v_s: -65.0,
            v_d: -65.0,
            tau_s: 20.0,
            tau_d: 50.0,
            g_c: 0.5,
            delta_t: 2.0,
            v_rest: -65.0,
            v_threshold: -50.0,
            v_peak: 20.0,
            v_reset: -65.0,
            dt: 0.1,
        }
    }

    fn valid(&self) -> bool {
        self.v_s.is_finite()
            && self.v_d.is_finite()
            && self.tau_s.is_finite()
            && self.tau_s > 0.0
            && self.tau_d.is_finite()
            && self.tau_d > 0.0
            && self.g_c.is_finite()
            && self.g_c >= 0.0
            && self.delta_t.is_finite()
            && self.delta_t > 0.0
            && self.v_rest.is_finite()
            && self.v_threshold.is_finite()
            && self.v_peak.is_finite()
            && self.v_reset.is_finite()
            && self.dt.is_finite()
            && self.dt > 0.0
    }

    fn derivatives(&self, vs: f64, vd: f64, current: f64) -> (f64, f64) {
        let vs_eff = vs.min(self.v_peak);
        let dvd = (-(vd - self.v_rest) + current - self.g_c * (vd - vs_eff)) / self.tau_d;
        let exp_arg = ((vs_eff - self.v_threshold) / self.delta_t).min(20.0);
        let exp_term = self.delta_t * exp_arg.exp();
        let dvs = (-(vs_eff - self.v_rest) + exp_term + self.g_c * (vd - vs_eff)) / self.tau_s;
        (dvs, dvd)
    }

    fn rk4_substep(&self, vs: f64, vd: f64, current: f64) -> (f64, f64) {
        let dt = self.dt;
        let (k1vs, k1vd) = self.derivatives(vs, vd, current);
        let (k2vs, k2vd) = self.derivatives(vs + 0.5 * dt * k1vs, vd + 0.5 * dt * k1vd, current);
        let (k3vs, k3vd) = self.derivatives(vs + 0.5 * dt * k2vs, vd + 0.5 * dt * k2vd, current);
        let (k4vs, k4vd) = self.derivatives(vs + dt * k3vs, vd + dt * k3vd, current);
        (
            vs + dt * (k1vs + 2.0 * k2vs + 2.0 * k3vs + k4vs) / 6.0,
            vd + dt * (k1vd + 2.0 * k2vd + 2.0 * k3vd + k4vd) / 6.0,
        )
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !i_ext.is_finite() || !self.valid() {
            return 0;
        }
        let (next_vs, next_vd) = self.rk4_substep(self.v_s, self.v_d, i_ext);
        if !next_vs.is_finite() || !next_vd.is_finite() {
            return 0;
        }
        self.v_d = next_vd;
        if next_vs >= self.v_peak {
            self.v_s = self.v_reset;
            1
        } else {
            self.v_s = next_vs;
            0
        }
    }

    pub fn reset(&mut self) {
        self.v_s = -65.0;
        self.v_d = -65.0;
    }
}

pub fn validate_neurogrid(state: &NeuroGridNeuron) -> bool {
    state.valid()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neurogrid_new() {
        let state = NeuroGridNeuron::new();
        assert!(validate_neurogrid(&state));
    }

    #[test]
    fn test_neurogrid_step() {
        let mut state = NeuroGridNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_neurogrid_cross_backend_anchor() {
        let mut state = NeuroGridNeuron::new();
        let mut spikes = 0_i32;
        for _ in 0..20_000 {
            spikes += state.step(100.0);
        }
        assert_eq!(spikes, 94);
        assert!(state.v_s.is_finite());
        assert!(state.v_d.is_finite());
    }

    #[test]
    fn test_neurogrid_invalid_input_preserves_state() {
        let mut state = NeuroGridNeuron::new();
        for _ in 0..10 {
            let _ = state.step(100.0);
        }
        let old = (state.v_s, state.v_d);
        assert_eq!(state.step(f64::INFINITY), 0);
        assert_eq!((state.v_s, state.v_d), old);
    }
}
