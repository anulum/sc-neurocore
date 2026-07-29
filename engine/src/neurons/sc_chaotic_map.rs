// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — preserved SC engineering two-state chaotic map

//! Project-designed map retained separately from the source Aihara model.

#[derive(Clone, Debug)]
pub struct SCChaoticMapNeuron {
    pub x: f64,
    pub y: f64,
    pub k_f: f64,
    pub k_s: f64,
    pub alpha: f64,
    pub delta: f64,
    pub x_threshold: f64,
}

impl Default for SCChaoticMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl SCChaoticMapNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            k_f: 0.7,
            k_s: 0.95,
            alpha: 2.0,
            delta: 0.05,
            x_threshold: 0.5,
        }
    }

    fn sigmoid(value: f64) -> f64 {
        if value >= 0.0 {
            1.0 / (1.0 + (-value).exp())
        } else {
            let exponential = value.exp();
            exponential / (1.0 + exponential)
        }
    }

    fn valid(&self) -> bool {
        [
            self.x,
            self.y,
            self.k_f,
            self.k_s,
            self.alpha,
            self.delta,
            self.x_threshold,
        ]
        .iter()
        .all(|value| value.is_finite())
            && self.k_f >= 0.0
            && self.delta >= 0.0
    }

    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !self.valid() || !current.is_finite() {
            return Err("invalid SC chaotic map state, parameters, or current");
        }
        let previous = self.x;
        let x_next = self.k_f * self.x * Self::sigmoid(self.x + self.alpha) - self.y + current;
        let y_next = self.k_s * self.y + self.delta * self.x;
        if !x_next.is_finite() || !y_next.is_finite() {
            return Err("non-finite SC chaotic map candidate");
        }
        self.x = x_next.clamp(-10.0, 10.0);
        self.y = y_next.clamp(-10.0, 10.0);
        Ok(i32::from(
            previous < self.x_threshold && self.x >= self.x_threshold,
        ))
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    pub fn reset(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
    }
}

#[derive(Clone, Debug)]
pub struct SCChaoticMapBatchResult {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub spikes: Vec<u8>,
    pub x_final: f64,
    pub y_final: f64,
    pub spike_count: usize,
}

pub fn simulate_sc_chaotic_map(
    x: f64,
    y: f64,
    k_f: f64,
    k_s: f64,
    alpha: f64,
    delta: f64,
    x_threshold: f64,
    current: &[f64],
) -> Result<SCChaoticMapBatchResult, &'static str> {
    if current.len() > i32::MAX as usize {
        return Err("current exceeds the signed-32-bit step limit");
    }
    let mut neuron = SCChaoticMapNeuron {
        x,
        y,
        k_f,
        k_s,
        alpha,
        delta,
        x_threshold,
    };
    if current.iter().any(|value| !value.is_finite()) {
        return Err("current must contain only finite values");
    }
    if !neuron.valid() {
        return Err("invalid SC chaotic map state or parameters");
    }

    let mut x_trace = Vec::with_capacity(current.len());
    let mut y_trace = Vec::with_capacity(current.len());
    let mut spikes = Vec::with_capacity(current.len());
    let mut spike_count = 0usize;
    for &drive in current {
        let event = neuron.try_step(drive)?;
        x_trace.push(neuron.x);
        y_trace.push(neuron.y);
        spikes.push(event as u8);
        spike_count += event as usize;
    }
    Ok(SCChaoticMapBatchResult {
        x: x_trace,
        y: y_trace,
        spikes,
        x_final: neuron.x,
        y_final: neuron.y,
        spike_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_recurrence_matches_independent_step() {
        let mut neuron = SCChaoticMapNeuron {
            x: 0.4,
            y: -0.2,
            ..Default::default()
        };
        let expected_x = 0.7 * 0.4 / (1.0 + (-2.4_f64).exp()) + 0.2 + 0.1;
        let expected_y = 0.95 * -0.2 + 0.05 * 0.4;
        neuron.try_step(0.1).unwrap();
        assert!((neuron.x - expected_x).abs() < 1.0e-15);
        assert!((neuron.y - expected_y).abs() < 1.0e-15);
    }

    #[test]
    fn rejected_input_is_atomic() {
        let mut neuron = SCChaoticMapNeuron::new();
        assert!(neuron.try_step(f64::NAN).is_err());
        assert_eq!((neuron.x, neuron.y), (0.0, 0.0));
    }

    #[test]
    fn batch_returns_complete_receipts() {
        let result =
            simulate_sc_chaotic_map(0.4, -0.2, 0.7, 0.95, 2.0, 0.05, 0.5, &[0.1, 0.1]).unwrap();
        assert_eq!(result.x.len(), 2);
        assert_eq!(result.y.len(), 2);
        assert_eq!(result.spikes, vec![1, 0]);
        assert_eq!(result.spike_count, 1);
    }
}
