// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Parametric LIF Neuron

#![deny(missing_docs)]

/// Parametric LIF — learnable decay via sigmoid(a). Fang et al. 2021.
#[derive(Clone, Debug)]
pub struct ParametricLIFNeuron {
    /// Membrane potential (dimensionless).
    pub v: f64,
    /// Trainable parameter setting the membrane decay (the leak is a function of a) (dimensionless).
    pub a: f64,
    /// Firing threshold (normalised) (dimensionless).
    pub threshold: f64,
}

impl ParametricLIFNeuron {
    /// Create a neuron at rest with decay parameter `a` and the given threshold.
    pub fn new(a: f64, threshold: f64) -> Self {
        Self {
            v: 0.0,
            a,
            threshold,
        }
    }

    /// Advance one step with the input `current` and return the spike of the updated `v`.
    ///
    /// The stored `v` is the previous charge `H[t-1]`; if it reached `threshold` it fired
    /// and was reset, so its decay term is dropped. The charge becomes
    /// `H[t] = sigmoid(a) * v * (1 - S[t-1]) + current`, and the returned spike is
    /// `S[t] = 1` if `H[t] >= threshold`, else 0 (Fang et al. 2021, Eq. 2-3), as in the
    /// Python model and the Go, Julia and safety-Rust lanes. The input is not validated.
    pub fn step(&mut self, current: f64) -> i32 {
        let alpha = self.alpha();
        let previous = if self.v >= self.threshold { 1.0 } else { 0.0 };
        self.v = alpha * self.v * (1.0 - previous) + current;
        if self.v >= self.threshold {
            1
        } else {
            0
        }
    }

    /// The membrane decay `sigmoid(a)`, in the sign-split form the Python model and the Go,
    /// Julia and safety-Rust lanes use: `1 / (1 + exp(-a))` for `a >= 0` and
    /// `exp(a) / (1 + exp(a))` below, which never overflows and gives the same bits as they do.
    pub fn alpha(&self) -> f64 {
        if self.a >= 0.0 {
            1.0 / (1.0 + (-self.a).exp())
        } else {
            let z = self.a.exp();
            z / (1.0 + z)
        }
    }

    /// Clear the membrane potential; the parameters are kept.
    pub fn reset(&mut self) {
        self.v = 0.0;
    }
}

impl Default for ParametricLIFNeuron {
    fn default() -> Self {
        Self::new(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plif_returns_the_spike_of_the_updated_potential() {
        // a = 0 gives alpha = 0.5. A first input of 5.0 lifts v from 0 to 5, over the
        // threshold, so the first step spikes; the next step drops the decay term and the
        // potential becomes the new input alone.
        let mut n = ParametricLIFNeuron::default();
        assert_eq!(n.step(5.0), 1);
        assert!((n.v - 5.0).abs() < 1e-15);
        assert_eq!(n.step(0.5), 0);
        assert!((n.v - 0.5).abs() < 1e-15);
        assert_eq!(n.step(0.75), 1);
        assert!((n.v - 1.0).abs() < 1e-15);
    }
    #[test]
    fn plif_fires() {
        let mut n = ParametricLIFNeuron::default();
        let total: i32 = (0..20).map(|_| n.step(1.5)).sum();
        assert!(total > 0);
    }
    #[test]
    fn plif_silent_without_input() {
        let mut n = ParametricLIFNeuron::default();
        let t: i32 = (0..100).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn plif_reset_clears_state() {
        let mut n = ParametricLIFNeuron::default();
        for _ in 0..20 {
            n.step(1.5);
        }
        n.reset();
        assert!((n.v - 0.0).abs() < 1e-10);
    }
    #[test]
    fn plif_bounded() {
        let mut n = ParametricLIFNeuron::default();
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn plif_alpha_takes_the_sign_split_form() {
        // Below zero the decay is exp(a) / (1 + exp(a)), not 1 / (1 + exp(-a)): at a = -3 the
        // two differ in the last bit, and the Python model and the other lanes use the former.
        let n = ParametricLIFNeuron::new(-3.0, 1.0);
        let z = (-3.0_f64).exp();
        assert_eq!(n.alpha().to_bits(), (z / (1.0 + z)).to_bits());
        assert_ne!(n.alpha().to_bits(), (1.0 / (1.0 + 3.0_f64.exp())).to_bits());
        let m = ParametricLIFNeuron::new(2.0, 1.0);
        assert_eq!(
            m.alpha().to_bits(),
            (1.0 / (1.0 + (-2.0_f64).exp())).to_bits()
        );
        let far = ParametricLIFNeuron::new(-800.0, 1.0);
        assert_eq!(far.alpha(), 0.0);
        assert!(ParametricLIFNeuron::new(800.0, 1.0).alpha() == 1.0);
    }
    #[test]
    fn plif_nan_no_panic() {
        ParametricLIFNeuron::default().step(f64::NAN);
    }
}
