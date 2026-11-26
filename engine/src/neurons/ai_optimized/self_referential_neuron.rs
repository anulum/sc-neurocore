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
    steps: u64,
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
            steps: 0,
        }
    }

    /// Advance one forward-Euler step and return 1 on a spike, else 0.
    ///
    /// The time constant is `tau * (1 + rate / target_rate)`, where `rate` is the number of
    /// spikes in the last 50 steps divided by the steps seen since the last reset, at most 50,
    /// so during the first 50 steps the rate is taken over the steps seen so far, as in the
    /// Python model. More firing slows the membrane; at the target rate the time constant is
    /// doubled. Reaching `theta` sets `v` to 0. The input is not validated.
    pub fn step(&mut self, current: f64) -> i32 {
        let n_spikes: u32 = self.history.iter().map(|&x| x as u32).sum();
        let seen = self.steps.min(self.window as u64);
        let rate = if seen == 0 {
            0.0
        } else {
            n_spikes as f64 / seen as f64
        };
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
        self.steps = self.steps.saturating_add(1);
        fired
    }

    /// Clear `v`, the spike history and the step count; the parameters are kept.
    pub fn reset(&mut self) {
        self.v = 0.0;
        self.history.fill(0);
        self.head = 0;
        self.steps = 0;
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
    fn self_referential_rate_is_taken_over_the_steps_seen_during_warm_up() {
        // Under a constant 2.0, v climbs as 2 * (1 - 0.9^k) and first reaches 1 at step 7.
        // At step 8 one spike in 7 steps gives rate 1/7, not 1/50, so tau_eff = 10 * (1 + 10/7).
        let mut n = SelfReferentialNeuron::new();
        let spikes: Vec<i32> = (0..7).map(|_| n.step(2.0)).collect();
        assert_eq!(spikes, vec![0, 0, 0, 0, 0, 0, 1]);
        assert_eq!(n.v, 0.0);
        n.step(2.0);
        let tau_eff = 10.0 * (1.0 + (1.0 / 7.0) / 0.1);
        assert_eq!(n.v, (-0.0 + 2.0) / tau_eff * 1.0);
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
