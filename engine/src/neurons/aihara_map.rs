// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Aihara chaotic map neuron

/// Aihara 1990 — chaotic neuron map with sigmoid nonlinearity.
///
/// 2D discrete map producing chaotic spiking, bursting, and tonic firing
/// depending on parameters. The sigmoid output function models the
/// nonlinear voltage-to-firing-rate relationship.
///
/// x(n+1) = k_f * x(n) / (1 + exp(-(x(n) + alpha))) - y(n) + I
/// y(n+1) = k_s * y(n) + delta * x(n)
///
/// Aihara et al., Phys Lett A 144:333, 1990.
#[derive(Clone, Debug)]
pub struct AiharaMapNeuron {
    pub x: f64,
    pub y: f64,
    pub k_f: f64,
    pub k_s: f64,
    pub alpha: f64,
    pub delta: f64,
    pub x_threshold: f64,
}

impl Default for AiharaMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl AiharaMapNeuron {
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

    pub fn step(&mut self, current: f64) -> i32 {
        let x_prev = self.x;
        let sigmoid = 1.0 / (1.0 + (-(self.x + self.alpha)).exp());
        let x_new = self.k_f * self.x * sigmoid - self.y + current;
        let y_new = self.k_s * self.y + self.delta * self.x;

        self.x = x_new.clamp(-10.0, 10.0);
        self.y = y_new.clamp(-10.0, 10.0);

        if !self.x.is_finite() {
            self.x = 0.0;
        }
        if !self.y.is_finite() {
            self.y = 0.0;
        }

        if self.x >= self.x_threshold && x_prev < self.x_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fires_with_input() {
        let mut neuron = AiharaMapNeuron::new();
        let spikes: i32 = (0..2000).map(|_| neuron.step(1.0)).sum();
        assert!(spikes > 0, "Aihara must fire with input, got {spikes}");
    }

    #[test]
    fn silent_without_input() {
        let mut neuron = AiharaMapNeuron::new();
        let spikes: i32 = (0..5000).map(|_| neuron.step(0.0)).sum();
        assert_eq!(
            spikes, 0,
            "Aihara must be silent without input, got {spikes}"
        );
    }

    #[test]
    fn chaotic_dynamics() {
        let mut neuron = AiharaMapNeuron::new();
        let mut values = Vec::new();
        for _ in 0..1000 {
            neuron.step(0.5);
            values.push(neuron.x);
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;
        assert!(variance > 0.001, "trajectory variance={variance}");
    }

    #[test]
    fn negative_input_stays_finite() {
        let mut neuron = AiharaMapNeuron::new();
        for _ in 0..10_000 {
            neuron.step(-100.0);
        }
        assert!(neuron.x.is_finite());
    }

    #[test]
    fn nan_input_stays_finite() {
        let mut neuron = AiharaMapNeuron::new();
        neuron.step(f64::NAN);
        assert!(neuron.x.is_finite());
    }

    #[test]
    fn extreme_input_is_bounded() {
        let mut neuron = AiharaMapNeuron::new();
        for _ in 0..1000 {
            neuron.step(1e6);
        }
        assert!(neuron.x.is_finite() && neuron.x <= 1e6);
    }

    #[test]
    fn reset_clears_state() {
        let mut neuron = AiharaMapNeuron::new();
        for _ in 0..100 {
            neuron.step(1.0);
        }
        neuron.reset();
        assert_eq!(neuron.x, 0.0);
        assert_eq!(neuron.y, 0.0);
    }

    #[test]
    fn rate_increases_with_input() {
        let mut low = AiharaMapNeuron::new();
        let mut high = AiharaMapNeuron::new();
        let spikes_low: i32 = (0..5000).map(|_| low.step(0.5)).sum();
        let spikes_high: i32 = (0..5000).map(|_| high.step(2.0)).sum();
        assert!(
            spikes_high >= spikes_low,
            "high={spikes_high}, low={spikes_low}"
        );
    }

    #[test]
    fn performance_100k_steps() {
        let start = std::time::Instant::now();
        let mut neuron = AiharaMapNeuron::new();
        for _ in 0..100_000 {
            std::hint::black_box(neuron.step(0.5));
        }
        assert!(start.elapsed().as_millis() < 50);
    }
}
