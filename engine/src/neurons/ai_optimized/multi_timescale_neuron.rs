// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Multi-timescale neuron model

/// Three-compartment memory neuron (fast/medium/slow timescales).
/// Slow compartment accumulates context, modulating excitability.
#[derive(Clone, Debug)]
pub struct MultiTimescaleNeuron {
    pub v_fast: f64,
    pub v_medium: f64,
    pub v_slow: f64,
    pub tau_fast: f64,
    pub tau_medium: f64,
    pub tau_slow: f64,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub theta_base: f64,
    pub dt: f64,
}

impl MultiTimescaleNeuron {
    pub fn new() -> Self {
        Self {
            v_fast: 0.0,
            v_medium: 0.0,
            v_slow: 0.0,
            tau_fast: 5.0,
            tau_medium: 200.0,
            tau_slow: 10000.0,
            alpha: 10.0,
            beta: 0.05,
            gamma: 0.3,
            theta_base: 1.0,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.v_fast += (-self.v_fast + current) / self.tau_fast * self.dt;
        let theta_eff = self.theta_base - self.gamma * self.v_slow;
        let fired = if self.v_fast >= theta_eff { 1 } else { 0 };
        self.v_medium += (-self.v_medium + self.alpha * fired as f64) / self.tau_medium * self.dt;
        self.v_slow += (-self.v_slow + self.beta * self.v_medium) / self.tau_slow * self.dt;
        if fired == 1 {
            self.v_fast = 0.0;
        }
        fired
    }

    pub fn reset(&mut self) {
        self.v_fast = 0.0;
        self.v_medium = 0.0;
        self.v_slow = 0.0;
    }
}

impl Default for MultiTimescaleNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_timescale_fires() {
        let mut n = MultiTimescaleNeuron::new();
        let total: i32 = (0..200).map(|_| n.step(2.0)).sum();
        assert!(total > 0);
    }

    #[test]
    fn multi_timescale_slow_accumulates() {
        let mut n = MultiTimescaleNeuron::new();
        for _ in 0..500 {
            n.step(2.0);
        }
        assert!(n.v_slow > 0.0);
    }

    #[test]
    fn multi_timescale_reset() {
        let mut n = MultiTimescaleNeuron::new();
        for _ in 0..100 {
            n.step(2.0);
        }
        n.reset();
        assert_eq!(n.v_fast, 0.0);
        assert_eq!(n.v_medium, 0.0);
        assert_eq!(n.v_slow, 0.0);
    }
}
