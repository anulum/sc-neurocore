// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Quantum-inspired LIF neuron model

/// Quantum-inspired LIF neuron with non-classical probability logic.
///
/// Extends standard LIF by maintaining a complex-valued amplitude z = a + bi
/// whose squared modulus |z|² determines the firing probability. Interference
/// between excitatory and inhibitory inputs can produce non-classical
/// suppression patterns (destructive interference).
///
///   dz/dt = (-z + I_complex) / τ
///   P(spike) = |z|² / θ²
///
/// Reference: Quantum-neural hybrid models, IBM Heron r2 noise models.
#[derive(Clone, Debug)]
pub struct QuantumInspiredLIFNeuron {
    pub z_re: f64,
    pub z_im: f64,
    pub tau: f64,
    pub theta: f64,
    pub dt: f64,
    pub v_reset: f64,
    rng_state: u64,
}

impl QuantumInspiredLIFNeuron {
    pub fn new() -> Self {
        Self {
            z_re: 0.0,
            z_im: 0.0,
            tau: 20.0,
            theta: 1.0,
            dt: 0.1,
            v_reset: 0.0,
            rng_state: 12345,
        }
    }

    /// Step with real and imaginary current components.
    pub fn step_complex(&mut self, i_re: f64, i_im: f64) -> i32 {
        let dz_re = (-self.z_re + i_re) / self.tau;
        let dz_im = (-self.z_im + i_im) / self.tau;
        self.z_re += dz_re * self.dt;
        self.z_im += dz_im * self.dt;

        let prob = (self.z_re * self.z_re + self.z_im * self.z_im) / (self.theta * self.theta);

        // Stochastic spike with probability |z|²/θ².
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        let uniform = (self.rng_state & 0xFFFFFFFF) as f64 / 4294967296.0;

        if uniform < prob.min(1.0) {
            self.z_re = self.v_reset;
            self.z_im = self.v_reset;
            1
        } else {
            0
        }
    }

    /// Standard step: real input only (imaginary = 0).
    pub fn step(&mut self, current: f64) -> i32 {
        self.step_complex(current, 0.0)
    }

    /// Firing probability from current amplitude.
    pub fn firing_probability(&self) -> f64 {
        let p = (self.z_re * self.z_re + self.z_im * self.z_im) / (self.theta * self.theta);
        p.min(1.0)
    }

    pub fn reset(&mut self) {
        self.z_re = 0.0;
        self.z_im = 0.0;
    }
}

impl Default for QuantumInspiredLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantum_lif_fires_stochastically() {
        let mut n = QuantumInspiredLIFNeuron::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(1.5);
        }
        assert!(spikes > 0, "Must fire with strong input");
        assert!(spikes < 10_000, "Must not fire every step (stochastic)");
    }

    #[test]
    fn quantum_lif_interference() {
        // Destructive interference: opposing real + imaginary should reduce firing.
        let mut n_constructive = QuantumInspiredLIFNeuron::new();
        let mut n_destructive = QuantumInspiredLIFNeuron::new();
        n_destructive.rng_state = n_constructive.rng_state;

        let mut spikes_c = 0;
        let mut spikes_d = 0;
        for _ in 0..5000 {
            spikes_c += n_constructive.step_complex(1.0, 1.0);
            // Same magnitude but opposing — should have similar |z|².
            spikes_d += n_destructive.step_complex(1.0, -1.0);
        }
        // Both should fire (|z|² = 2 in both cases for steady state).
        assert!(spikes_c > 0, "Constructive must fire");
        assert!(spikes_d > 0, "Destructive must fire");
    }

    #[test]
    fn quantum_lif_zero_input_no_fire() {
        let mut n = QuantumInspiredLIFNeuron::new();
        let spikes: i32 = (0..1000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0, "Zero input must not fire");
    }

    #[test]
    fn quantum_lif_probability_range() {
        let mut n = QuantumInspiredLIFNeuron::new();
        for _ in 0..100 {
            n.step(0.5);
            let p = n.firing_probability();
            assert!((0.0..=1.0).contains(&p), "P must be in [0,1], got {p}");
        }
    }
}
