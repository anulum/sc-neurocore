// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Compositional-binding neuron model

#![deny(missing_docs)]

/// Phase-coding neuron for compositional variable binding.
/// Spike when amplitude * cos(phase) > threshold, on every step while it holds.
/// The phase advances freely: there is no input phase and `coupling` is not used.
#[derive(Clone, Debug)]
pub struct CompositionalBindingNeuron {
    /// Phase (rad); it advances by `omega * dt` each step and is not wrapped.
    pub phi: f64,
    /// Amplitude (dimensionless); it relaxes towards the input with `tau`.
    pub amplitude: f64,
    /// Intrinsic phase oscillation frequency (rad/ms).
    pub omega: f64,
    /// Phase-coupling strength to the input oscillation (dimensionless). No step reads it:
    /// the phase has no input term.
    pub coupling: f64,
    /// Amplitude time constant (ms).
    pub tau: f64,
    /// Firing threshold on `amplitude * cos(phi)` (dimensionless).
    pub theta: f64,
    /// Step (ms).
    pub dt: f64,
}

impl CompositionalBindingNeuron {
    /// Create a neuron at phase 0 and amplitude 0 with the default parameters and
    /// `dt` = 1 ms.
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

    /// Advance one step and return 1 when `amplitude * cos(phi)` exceeds `theta`, else 0.
    ///
    /// The phase advances by `omega * dt`, and the amplitude takes a forward-Euler step
    /// towards `current` with `tau`. Nothing is reset by an event. The input is not
    /// validated.
    pub fn step(&mut self, current: f64) -> i32 {
        self.phi += self.omega * self.dt;
        self.amplitude += (-self.amplitude + current) / self.tau * self.dt;
        if self.amplitude * self.phi.cos() > self.theta {
            1
        } else {
            0
        }
    }

    /// Return the phase and the amplitude to 0; the parameters are kept.
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
