// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety mirror of the sliding PSN

//! k-order sliding Parallel Spiking Neuron — Fang et al. (2023).
//!
//! Standalone mirror of the Python reference:
//! `H[t] = sum_{i=0}^{k-1} W_i * X[t-k+1+i]` with zero pre-history,
//! `S[t] = Theta(H[t] - v_threshold)`, `Theta(0) = 1`, no reset on
//! firing. The sum accumulates sequentially from i = 0 so the result
//! is bit-for-bit identical to every other backend.

#[derive(Debug, Clone)]
pub struct ParallelSpikingNeuron {
    pub weights: Vec<f64>,
    pub history: Vec<f64>,
    pub v_threshold: f64,
    pub hidden: f64,
}

impl ParallelSpikingNeuron {
    pub fn new() -> Self {
        Self::with_kernel(8, 1.0)
    }

    pub fn with_kernel(kernel_size: usize, v_threshold: f64) -> Self {
        let k = kernel_size.max(1);
        Self {
            weights: vec![1.0 / k as f64; k],
            history: vec![0.0; k],
            v_threshold,
            hidden: 0.0,
        }
    }

    fn valid(&self) -> bool {
        !self.weights.is_empty()
            && self.history.len() == self.weights.len()
            && self.weights.iter().all(|w| w.is_finite())
            && self.history.iter().all(|x| x.is_finite())
            && self.v_threshold.is_finite()
    }

    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() {
            return Err("current must be finite");
        }
        if !self.valid() {
            return Err("sliding PSN state and parameters must be finite");
        }

        let mut hidden = 0.0_f64;
        for (index, weight) in self.weights.iter().enumerate() {
            let value = if index + 1 < self.history.len() {
                self.history[index + 1]
            } else {
                current
            };
            hidden += weight * value;
        }
        if !hidden.is_finite() {
            return Err("sliding PSN hidden state became non-finite");
        }

        self.history.rotate_left(1);
        if let Some(last) = self.history.last_mut() {
            *last = current;
        }
        self.hidden = hidden;
        Ok(if hidden >= self.v_threshold { 1 } else { 0 })
    }

    pub fn reset(&mut self) {
        self.history.fill(0.0);
        self.hidden = 0.0;
    }
}

impl Default for ParallelSpikingNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_psn(state: &ParallelSpikingNeuron) -> bool {
    !state.weights.is_empty()
        && state.history.len() == state.weights.len()
        && state.weights.iter().all(|w| w.is_finite())
        && state.history.iter().all(|x| x.is_finite())
        && state.v_threshold.is_finite()
        && state.hidden.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_paper_equation_oracle() {
        let drive: Vec<f64> = (0..32)
            .map(|i| 0.4 + 0.3 * (i as f64 * 0.17).sin())
            .collect();
        let weights = [0.1, -0.2, 0.35, 0.75];
        let mut n = ParallelSpikingNeuron::with_kernel(4, 0.4);
        n.weights = weights.to_vec();
        for (t, &current) in drive.iter().enumerate() {
            let spike = n.step(current).expect("finite configured drive");
            let mut hidden = 0.0_f64;
            for (i, w) in weights.iter().enumerate() {
                let j = t as i64 - 3 + i as i64;
                let x = if j < 0 { 0.0 } else { drive[j as usize] };
                hidden += w * x;
            }
            assert_eq!(n.hidden.to_bits(), hidden.to_bits());
            assert_eq!(spike, i32::from(hidden >= 0.4));
        }
        assert!(validate_psn(&n));
    }

    #[test]
    fn firing_never_clears_history_and_theta_is_right_continuous() {
        let mut n = ParallelSpikingNeuron::with_kernel(1, 1.0);
        assert_eq!(n.step(1.0).expect("finite drive"), 1);
        assert_eq!(n.history, vec![1.0]);
    }

    #[test]
    fn invalid_input_is_rejected_atomically() {
        let mut n = ParallelSpikingNeuron::new();
        n.step(0.7).expect("finite drive");
        let before = (n.history.clone(), n.hidden);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(n.step(bad).is_err());
        }
        assert_eq!((n.history.clone(), n.hidden), before);
    }

    #[test]
    fn reset_clears_history_only() {
        let mut n = ParallelSpikingNeuron::new();
        n.step(1.0).expect("finite drive");
        n.reset();
        assert!(n.history.iter().all(|&x| x == 0.0));
        assert_eq!(n.hidden, 0.0);
        assert_eq!(n.weights, vec![0.125; 8]);
    }
}
