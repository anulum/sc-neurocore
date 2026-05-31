// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for ermentrout_kopell_map_neuron

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct ErmentroutKopellMapNeuron {
    pub theta: f64,
    pub dt: f64,
    pub gain: f64,
    pub theta_threshold: f64,
}

impl ErmentroutKopellMapNeuron {
    pub fn new() -> Self {
        Self {
            theta: 0.0_f64,
            dt: 0.1_f64,
            gain: 1.0_f64,
            theta_threshold: std::f64::consts::PI,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !validate_ermentrout_kopell_map_neuron(self) {
            return Err("invalid Ermentrout-Kopell runtime state");
        }
        if !i_ext.is_finite() {
            return Err("invalid Ermentrout-Kopell current");
        }

        let inp = self.gain * i_ext;
        if !inp.is_finite() {
            return Err("invalid Ermentrout-Kopell input drive");
        }
        let theta_prev = self.theta;
        let cos_theta = self.theta.cos();
        let d_theta = (1.0 - cos_theta) + (1.0 + cos_theta) * inp;
        let theta_next = self.theta + self.dt * d_theta;
        if !d_theta.is_finite() || !theta_next.is_finite() {
            return Err("invalid Ermentrout-Kopell candidate phase");
        }
        let fired = if theta_next >= self.theta_threshold && theta_prev < self.theta_threshold {
            1
        } else {
            0
        };
        self.theta = theta_next.rem_euclid(2.0 * std::f64::consts::PI);
        Ok(fired)
    }

    pub fn reset(&mut self) {
        // self.theta = 0.0
        self.theta = 0.0_f64;
        self.dt = 0.1_f64;
        self.gain = 1.0_f64;
        self.theta_threshold = std::f64::consts::PI;
    }
}

pub fn validate_ermentrout_kopell_map_neuron(state: &ErmentroutKopellMapNeuron) -> bool {
    state.theta.is_finite()
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.gain.is_finite()
        && state.theta_threshold.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ermentrout_kopell_map_neuron_new() {
        let state = ErmentroutKopellMapNeuron::new();
        assert!(validate_ermentrout_kopell_map_neuron(&state));
    }

    #[test]
    fn test_ermentrout_kopell_map_neuron_step() {
        let mut state = ErmentroutKopellMapNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_ermentrout_kopell_map_neuron_rejects_invalid_runtime_state() {
        let mut state = ErmentroutKopellMapNeuron::new();
        state.theta = f64::INFINITY;
        assert!(state.step(1.0).is_err());
    }
}
