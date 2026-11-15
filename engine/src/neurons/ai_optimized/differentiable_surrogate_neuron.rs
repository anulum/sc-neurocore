// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Differentiable-surrogate neuron model

#![deny(missing_docs)]

/// Spiking neuron with learnable surrogate gradient parameters.
/// alpha (decay), beta (steepness), theta (threshold) are the parameters a trainer
/// would adjust; this type computes the forward step and does not adjust them.
#[derive(Clone, Debug)]
pub struct DifferentiableSurrogateNeuron {
    /// Membrane potential (dimensionless).
    pub v: f64,
    /// Per-step membrane decay factor; the decay term is dropped after a spike (dimensionless).
    pub alpha: f64,
    /// Surrogate-gradient sharpness (pseudo-derivative steepness) (dimensionless).
    pub beta: f64,
    /// Firing threshold (dimensionless).
    pub theta: f64,
}

impl DifferentiableSurrogateNeuron {
    /// Create a neuron at rest with `alpha` = 0.9, `beta` = 5 and threshold 1.
    pub fn new() -> Self {
        Self {
            v: 0.0,
            alpha: 0.9,
            beta: 5.0,
            theta: 1.0,
        }
    }

    /// Advance one step with `current` and return the spike of the stored `v`.
    ///
    /// The spike is read before the update: 1 if the stored `v` has reached `theta`. Then
    /// `v` becomes `alpha * v + current`, with the decay term dropped when that spike fired.
    /// The input is not validated.
    pub fn step(&mut self, current: f64) -> i32 {
        let spike = if self.v >= self.theta { 1 } else { 0 };
        self.v = self.alpha * self.v * (1.0 - spike as f64) + current;
        spike
    }

    /// Return the surrogate gradient `1 / (1 + beta * |v - theta|)^2` at the present `v`.
    pub fn surrogate_grad(&self) -> f64 {
        let d = (self.v - self.theta).abs();
        1.0 / ((1.0 + self.beta * d) * (1.0 + self.beta * d))
    }

    /// Clear the membrane potential; the parameters are kept.
    pub fn reset(&mut self) {
        self.v = 0.0;
    }
}

impl Default for DifferentiableSurrogateNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn differentiable_surrogate_fires() {
        let mut n = DifferentiableSurrogateNeuron::new();
        let total: i32 = (0..20).map(|_| n.step(1.5)).sum();
        assert!(total > 0);
    }

    #[test]
    fn differentiable_surrogate_grad_positive() {
        let n = DifferentiableSurrogateNeuron::new();
        assert!(n.surrogate_grad() > 0.0);
    }

    #[test]
    fn differentiable_surrogate_reset() {
        let mut n = DifferentiableSurrogateNeuron::new();
        for _ in 0..10 {
            n.step(1.5);
        }
        n.reset();
        assert_eq!(n.v, 0.0);
    }
}
