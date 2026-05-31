// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for aihara_map_neuron

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct AiharaMapNeuron {
    pub x: f64,
    pub y: f64,
    pub k_f: f64,
    pub k_s: f64,
    pub alpha: f64,
    pub delta: f64,
    pub x_threshold: f64,
}

impl AiharaMapNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0_f64,
            y: 0.0_f64,
            k_f: 0.7_f64,
            k_s: 0.95_f64,
            alpha: 2.0_f64,
            delta: 0.05_f64,
            x_threshold: 0.5_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !validate_aihara_map_neuron(self) {
            return Err("invalid Aihara map runtime state");
        }
        if !i_ext.is_finite() {
            return Err("invalid Aihara map current");
        }

        let x_prev = self.x;
        let sigmoid = logistic(self.x + self.alpha);
        let x_new = self.k_f * self.x * sigmoid - self.y + i_ext;
        let y_new = self.k_s * self.y + self.delta * self.x;
        if !x_new.is_finite() || !y_new.is_finite() {
            return Err("invalid Aihara map candidate state");
        }
        self.x = x_new.clamp(-10.0, 10.0);
        self.y = y_new.clamp(-10.0, 10.0);
        Ok(if self.x >= self.x_threshold && x_prev < self.x_threshold {
            1
        } else {
            0
        })
    }

    pub fn reset(&mut self) {
        // self.x = 0.0
        // self.y = 0.0
        self.x = 0.0_f64;
        self.y = 0.0_f64;
        self.k_f = 0.7_f64;
        self.k_s = 0.95_f64;
        self.alpha = 2.0_f64;
    }
}

pub fn validate_aihara_map_neuron(state: &AiharaMapNeuron) -> bool {
    state.x.is_finite()
        && state.y.is_finite()
        && state.k_f.is_finite()
        && state.k_f >= 0.0
        && state.k_s.is_finite()
        && state.alpha.is_finite()
        && state.delta.is_finite()
        && state.delta >= 0.0
        && state.x_threshold.is_finite()
}

fn logistic(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let exp_z = z.exp();
        exp_z / (1.0 + exp_z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aihara_map_neuron_new() {
        let state = AiharaMapNeuron::new();
        assert!(validate_aihara_map_neuron(&state));
    }

    #[test]
    fn test_aihara_map_neuron_step() {
        let mut state = AiharaMapNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_aihara_map_neuron_rejects_invalid_runtime_state() {
        let mut state = AiharaMapNeuron::new();
        state.y = f64::INFINITY;
        assert!(state.step(0.0).is_err());
    }
}
