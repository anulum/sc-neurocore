// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rulkov discrete map neuron

//! Rulkov discrete map neuron.

/// Rulkov 2001 — piecewise nonlinear map for fast/slow bursting.
#[derive(Clone, Debug)]
pub struct RulkovMapNeuron {
    pub x: f64,
    pub y: f64,
    pub alpha: f64,
    pub sigma: f64,
    pub mu: f64,
    pub x_threshold: f64,
}

impl RulkovMapNeuron {
    pub fn new() -> Self {
        Self {
            x: -1.0,
            y: -3.0,
            alpha: 4.0,
            sigma: -1.6,
            mu: 0.001,
            x_threshold: 0.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let x_prev = self.x;
        let x_new = if self.x <= 0.0 {
            self.alpha / (1.0 - self.x) + self.y + current
        } else if self.x < self.alpha + self.y + current {
            self.alpha + self.y + current
        } else {
            -1.0
        };
        let y_new = self.y - self.mu * (self.x + 1.0) + self.mu * self.sigma;
        self.x = x_new;
        self.y = y_new;
        if self.x >= self.x_threshold && x_prev < self.x_threshold {
            1
        } else {
            0
        }
    }
    /// Run `n_steps` under a constant input, returning the `x` trace and the
    /// upward-crossing spike count. Reuses `step` so the trace is bit-identical
    /// to the per-step path and to the Python reference. The final state is
    /// left in `self.x` / `self.y`.
    pub fn simulate(&mut self, n_steps: usize, current: f64) -> (Vec<f64>, i64) {
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes: i64 = 0;
        for _ in 0..n_steps {
            let spiked = self.step(current);
            trace.push(self.x);
            spikes += spiked as i64;
        }
        (trace, spikes)
    }
    pub fn reset(&mut self) {
        self.x = -1.0;
        self.y = -3.0;
    }
}
impl Default for RulkovMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rulkov_fires() {
        let mut n = RulkovMapNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(0.5)).sum();
        assert!(t > 0);
    }
}
