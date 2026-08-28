// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SC clipped-logistic discrete map neuron

//! Retained project-defined clipped-logistic fast/slow map.

/// Retained two-state clipped-logistic fast/slow recurrence.
#[derive(Clone, Debug)]
pub struct SCClippedLogisticBurstingMapNeuron {
    pub x: f64,
    pub y: f64,
    pub a: f64,
    pub epsilon: f64,
    pub sigma: f64,
    pub x_threshold: f64,
}

impl SCClippedLogisticBurstingMapNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.1,
            y: 0.0,
            a: 3.8,
            epsilon: 0.01,
            sigma: 0.5,
            x_threshold: 0.9,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let f = self.a * self.x * (1.0 - self.x);
        let x_new = (f - self.y + current).clamp(-2.0, 2.0);
        let y_new = self.y + self.epsilon * (self.x - self.sigma);
        self.x = x_new;
        self.y = y_new;
        if self.x >= self.x_threshold {
            1
        } else {
            0
        }
    }
    /// Run `n_steps` under a constant input, returning the `x` trace and the
    /// spike count. Reuses `step` so the trace is bit-identical to the
    /// per-step path and to the Python reference. The final state is left in
    /// `self.x` / `self.y`.
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
        self.x = 0.1;
        self.y = 0.0;
    }
}
impl Default for SCClippedLogisticBurstingMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_clipped_logistic_fires() {
        let mut n = SCClippedLogisticBurstingMapNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert!(t > 0);
    }
}
