// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Predictive-coding neuron model

/// Fires only on prediction errors. Silent when input matches prediction.
#[derive(Clone, Debug)]
pub struct PredictiveCodingNeuron {
    pub v: f64,
    pub pred: f64,
    pub tau: f64,
    pub tau_pred: f64,
    pub theta: f64,
    pub dt: f64,
}

impl PredictiveCodingNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0,
            pred: 0.0,
            tau: 10.0,
            tau_pred: 50.0,
            theta: 1.0,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let surprise = (current - self.pred).abs();
        self.pred += (current - self.pred) / self.tau_pred * self.dt;
        self.v += (-self.v + surprise) / self.tau * self.dt;
        if self.v >= self.theta {
            self.v = 0.0;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = 0.0;
        self.pred = 0.0;
    }
}

impl Default for PredictiveCodingNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictive_coding_fires_on_change() {
        let mut n = PredictiveCodingNeuron::new();
        for _ in 0..200 {
            n.step(1.0);
        }
        let spikes_after_change: i32 = (0..50).map(|_| n.step(10.0)).sum();
        assert!(spikes_after_change > 0);
    }

    #[test]
    fn predictive_coding_silent_on_constant() {
        let mut n = PredictiveCodingNeuron::new();
        for _ in 0..500 {
            n.step(0.5);
        }
        let late: i32 = (0..100).map(|_| n.step(0.5)).sum();
        assert_eq!(late, 0);
    }

    #[test]
    fn predictive_coding_reset() {
        let mut n = PredictiveCodingNeuron::new();
        for _ in 0..50 {
            n.step(5.0);
        }
        n.reset();
        assert_eq!(n.v, 0.0);
        assert_eq!(n.pred, 0.0);
    }
}
