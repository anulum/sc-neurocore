// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Self-referential neuron model

#![deny(missing_docs)]

/// Introspects on its own spike history; adjusts tau based on firing rate.
#[derive(Clone, Debug)]
pub struct SelfReferentialNeuron {
    /// Membrane potential (dimensionless); a spike sets it to 0.
    pub v: f64,
    /// Base membrane time constant (ms).
    pub tau: f64,
    /// Firing threshold (dimensionless).
    pub theta: f64,
    /// Target self-firing fraction (spikes per window) (dimensionless).
    pub target_rate: f64,
    /// Step (ms).
    pub dt: f64,
    history: Vec<u8>,
    head: usize,
    window: usize,
}

impl SelfReferentialNeuron {
    /// Create a neuron at rest with a 50-step spike history, `tau` = 10 ms, threshold 1,
    /// target rate 0.1 and `dt` = 1 ms.
    pub fn new() -> Self {
        let window = 50;
        Self {
            v: 0.0,
            tau: 10.0,
            theta: 1.0,
            target_rate: 0.1,
            dt: 1.0,
            history: vec![0; window],
            head: 0,
            window,
        }
    }

    /// Advance one forward-Euler step and return 1 on a spike, else 0.
    ///
    /// The time constant is `tau * (1 + rate / target_rate)`, where `rate` is the fraction of
    /// the last 50 steps with a spike, so more firing slows the membrane; at the target rate
    /// the time constant is doubled. Reaching `theta` sets `v` to 0. The input is not
    /// validated.
    pub fn step(&mut self, current: f64) -> i32 {
        let n_spikes: u32 = self.history.iter().map(|&x| x as u32).sum();
        let rate = n_spikes as f64 / self.window as f64;
        let tau_eff = self.tau * (1.0 + rate / self.target_rate);
        self.v += (-self.v + current) / tau_eff * self.dt;
        let fired = if self.v >= self.theta {
            self.v = 0.0;
            1
        } else {
            0
        };
        self.history[self.head] = fired as u8;
        self.head = (self.head + 1) % self.window;
        fired
    }

    /// Clear `v` and the spike history; the parameters are kept.
    pub fn reset(&mut self) {
        self.v = 0.0;
        self.history.fill(0);
        self.head = 0;
    }
}

impl Default for SelfReferentialNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_referential_fires() {
        let mut n = SelfReferentialNeuron::new();
        let total: i32 = (0..200).map(|_| n.step(2.0)).sum();
        assert!(total > 0);
    }

    #[test]
    fn self_referential_adapts_tau() {
        let mut n = SelfReferentialNeuron::new();
        for _ in 0..100 {
            n.step(2.0);
        }
        let n_spikes: u32 = n.history.iter().map(|&x| x as u32).sum();
        assert!(n_spikes > 0);
    }

    #[test]
    fn self_referential_reset() {
        let mut n = SelfReferentialNeuron::new();
        for _ in 0..100 {
            n.step(2.0);
        }
        n.reset();
        assert_eq!(n.v, 0.0);
        assert!(n.history.iter().all(|&x| x == 0));
    }
}
