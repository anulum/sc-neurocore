// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Compositional-binding neuron model

/// Phase-coding neuron for compositional variable binding.
/// Spike when amplitude * cos(phase) > threshold.
#[derive(Clone, Debug)]
pub struct CompositionalBindingNeuron {
    pub phi: f64,
    pub amplitude: f64,
    pub omega: f64,
    pub coupling: f64,
    pub tau: f64,
    pub theta: f64,
    pub dt: f64,
}

impl CompositionalBindingNeuron {
    pub fn new() -> Self {
        Self {
            phi: 0.0,
            amplitude: 0.0,
            omega: 0.1,
            coupling: 0.5,
            tau: 10.0,
            theta: 0.8,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.phi += self.omega * self.dt;
        self.amplitude += (-self.amplitude + current) / self.tau * self.dt;
        if self.amplitude * self.phi.cos() > self.theta {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.phi = 0.0;
        self.amplitude = 0.0;
    }
}

impl Default for CompositionalBindingNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compositional_binding_fires() {
        let mut n = CompositionalBindingNeuron::new();
        let total: i32 = (0..200).map(|_| n.step(2.0)).sum();
        assert!(total > 0);
    }

    #[test]
    fn compositional_binding_phase_advances() {
        let mut n = CompositionalBindingNeuron::new();
        for _ in 0..100 {
            n.step(1.0);
        }
        assert!(n.phi > 0.0);
    }

    #[test]
    fn compositional_binding_reset() {
        let mut n = CompositionalBindingNeuron::new();
        for _ in 0..100 {
            n.step(2.0);
        }
        n.reset();
        assert_eq!(n.phi, 0.0);
        assert_eq!(n.amplitude, 0.0);
    }
}
