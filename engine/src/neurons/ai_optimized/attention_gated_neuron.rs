// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Attention-gated neuron model

#![deny(missing_docs)]

/// Spiking neuron whose input passes a sigmoid attention gate.
/// The key and query weights are parameters; this type does not change them.
/// gate = sigmoid(w_key * I + w_query * v), modulates input before integration.
#[derive(Clone, Debug)]
pub struct AttentionGatedNeuron {
    /// Membrane potential (dimensionless); a spike sets it to 0.
    pub v: f64,
    /// Attention-gate weight on the input current (dimensionless).
    pub w_key: f64,
    /// Attention-gate weight on the membrane potential (dimensionless).
    pub w_query: f64,
    /// Membrane time constant (ms).
    pub tau: f64,
    /// Firing threshold (dimensionless).
    pub theta: f64,
    /// Step (ms).
    pub dt: f64,
}

impl AttentionGatedNeuron {
    /// Create a neuron at rest with the default gate weights, `tau` = 10 ms, threshold 1
    /// and `dt` = 1 ms.
    pub fn new() -> Self {
        Self {
            v: 0.0,
            w_key: 1.0,
            w_query: 0.5,
            tau: 10.0,
            theta: 1.0,
            dt: 1.0,
        }
    }

    /// Advance one forward-Euler step and return 1 on a spike, else 0.
    ///
    /// The gate `sigmoid(w_key * current + w_query * v)`, taken with the old `v`, scales the
    /// input, and `v` gains `(-v + gate * current) / tau * dt`. Reaching `theta` sets `v` to
    /// 0. The input is not validated.
    pub fn step(&mut self, current: f64) -> i32 {
        let gate = 1.0 / (1.0 + (-(self.w_key * current + self.w_query * self.v)).exp());
        self.v += (-self.v + gate * current) / self.tau * self.dt;
        if self.v >= self.theta {
            self.v = 0.0;
            1
        } else {
            0
        }
    }

    /// Clear the membrane potential; the parameters are kept.
    pub fn reset(&mut self) {
        self.v = 0.0;
    }
}

impl Default for AttentionGatedNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_gated_fires() {
        let mut n = AttentionGatedNeuron::new();
        let total: i32 = (0..200).map(|_| n.step(2.0)).sum();
        assert!(total > 0);
    }

    #[test]
    fn attention_gated_gate_suppresses_low_input() {
        let mut n = AttentionGatedNeuron {
            w_key: -2.0,
            ..AttentionGatedNeuron::new()
        };
        let total: i32 = (0..200).map(|_| n.step(0.1)).sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn attention_gated_reset() {
        let mut n = AttentionGatedNeuron::new();
        for _ in 0..50 {
            n.step(2.0);
        }
        n.reset();
        assert_eq!(n.v, 0.0);
    }
}
