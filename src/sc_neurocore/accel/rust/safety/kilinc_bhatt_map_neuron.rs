// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for kilinc_bhatt_map_neuron

#![allow(dead_code)]

#[derive(Debug, Clone)]
pub struct KilincBhattMapNeuron {
    pub x: f64,
    pub theta: f64,
    pub k: f64,
    pub beta: f64,
    pub gamma: f64,
    pub theta_spike: f64,
    pub x_threshold: f64,
}

impl KilincBhattMapNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0_f64,
            theta: 0.0_f64,
            k: 1.5_f64,
            beta: 0.95_f64,
            gamma: 0.3_f64,
            theta_spike: 0.8_f64,
            x_threshold: 0.8_f64,
        }
    }

    fn sigmoid(z: f64) -> f64 {
        if z >= 0.0 {
            1.0 / (1.0 + (-z).exp())
        } else {
            let exp_z = z.exp();
            exp_z / (1.0 + exp_z)
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !i_ext.is_finite() {
            return Err("current must be finite");
        }
        if !validate_kilinc_bhatt_map_neuron(self) {
            return Err("Kilinc-Bhatt state and parameters must satisfy the public bounds");
        }

        let x_prev = self.x;
        let sig = Self::sigmoid((self.x - self.theta) * 4.0);
        let x_new = -self.x + self.k * sig + i_ext;
        let spiked = if self.x >= self.theta_spike { 1.0 } else { 0.0 };
        let theta_new = self.beta * self.theta + self.gamma * spiked;
        if !x_new.is_finite() || !theta_new.is_finite() {
            return Err("Kilinc-Bhatt candidate state became non-finite");
        }

        self.x = x_new.clamp(-5.0, 5.0);
        self.theta = theta_new.clamp(-5.0, 5.0);
        Ok(if self.x >= self.x_threshold && x_prev < self.x_threshold {
            1
        } else {
            0
        })
    }

    pub fn reset(&mut self) {
        // self.x = 0.0
        // self.theta = 0.0
        self.x = 0.0_f64;
        self.theta = 0.0_f64;
    }
}

pub fn validate_kilinc_bhatt_map_neuron(state: &KilincBhattMapNeuron) -> bool {
    state.x.is_finite()
        && (-5.0..=5.0).contains(&state.x)
        && state.theta.is_finite()
        && (-5.0..=5.0).contains(&state.theta)
        && state.k.is_finite()
        && (0.0..=5.0).contains(&state.k)
        && state.beta.is_finite()
        && (0.0..=1.0).contains(&state.beta)
        && state.gamma.is_finite()
        && (0.0..=2.0).contains(&state.gamma)
        && state.theta_spike.is_finite()
        && (0.0..=2.0).contains(&state.theta_spike)
        && state.x_threshold.is_finite()
        && (0.0..=2.0).contains(&state.x_threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kilinc_bhatt_map_neuron_new() {
        let state = KilincBhattMapNeuron::new();
        assert!(validate_kilinc_bhatt_map_neuron(&state));
    }

    #[test]
    fn test_kilinc_bhatt_map_neuron_step() {
        let mut state = KilincBhattMapNeuron::new();
        let spike = state.step(10.0).expect("finite drive");
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_non_finite_drive_is_atomic() {
        let mut state = KilincBhattMapNeuron::new();
        let before = (state.x, state.theta);
        assert_eq!(state.step(f64::NAN), Err("current must be finite"));
        assert_eq!((state.x, state.theta), before);
    }

    #[test]
    fn test_invalid_state_is_atomic() {
        let mut state = KilincBhattMapNeuron::new();
        state.beta = 1.1;
        let before = (state.x, state.theta);
        assert!(state.step(1.0).is_err());
        assert_eq!((state.x, state.theta), before);
    }

    #[test]
    fn test_reset_preserves_parameters() {
        let mut state = KilincBhattMapNeuron::new();
        state.k = 2.0;
        state.step(1.0).expect("valid drive");
        state.reset();
        assert_eq!((state.x, state.theta, state.k), (0.0, 0.0, 2.0));
    }
}
