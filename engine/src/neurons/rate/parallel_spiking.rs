// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Parallel spiking neuron model

//! k-order sliding Parallel Spiking Neuron — Fang et al. (2023).

/// k-order sliding PSN — Fang et al. (2023), NeurIPS.
///
/// Streaming form of the PSN family (paper Eqs. 14–15):
///
/// ```text
/// H[t] = sum_{i=0}^{k-1} W_i * X[t-k+1+i],  X[j] = 0 for j < 0
/// S[t] = Theta(H[t] - v_threshold)
/// ```
///
/// `weights[k-1]` multiplies the newest input; the sum accumulates
/// sequentially from i = 0 so every backend reproduces the same
/// binary64 result bit-for-bit. `Theta(0) = 1` per the paper. No PSN
/// variant has a reset: firing never clears the input history. The
/// paper trains `W` and `v_threshold`; the uniform `1/k` defaults are
/// repository defaults.
#[derive(Clone, Debug)]
pub struct ParallelSpikingNeuron {
    pub weights: Vec<f64>,
    pub history: Vec<f64>,
    pub v_threshold: f64,
    pub hidden: f64,
}

impl ParallelSpikingNeuron {
    /// Construct with the uniform repository default weights `1/k`.
    pub fn new(kernel_size: usize, v_threshold: f64) -> Self {
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

    /// Advance one step after validating the input and configuration.
    ///
    /// Computes the hidden state on a candidate window and commits only
    /// on success: a non-finite input, an invalid configuration, or a
    /// non-finite hidden state returns `Err` with the pre-step state
    /// preserved exactly.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
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

    /// Fail-closed wrapper for legacy callers: returns 0 on any rejected
    /// input without mutating state.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Clear the retained inputs, preserving weights and threshold.
    pub fn reset(&mut self) {
        self.history.fill(0.0);
        self.hidden = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent oracle: explicit zero-padded window per paper Eq. 14.
    fn oracle(weights: &[f64], v_th: f64, drive: &[f64]) -> (Vec<f64>, Vec<i32>) {
        let k = weights.len();
        let mut hidden_trace = Vec::new();
        let mut spikes = Vec::new();
        for t in 0..drive.len() {
            let mut hidden = 0.0_f64;
            for (i, w) in weights.iter().enumerate() {
                let j = t as i64 - k as i64 + 1 + i as i64;
                let x = if j < 0 { 0.0 } else { drive[j as usize] };
                hidden += w * x;
            }
            hidden_trace.push(hidden);
            spikes.push(if hidden >= v_th { 1 } else { 0 });
        }
        (hidden_trace, spikes)
    }

    #[test]
    fn matches_paper_equation_oracle_bit_exactly() {
        let drive: Vec<f64> = (0..64)
            .map(|i| 0.4 + 0.3 * (i as f64 * 0.17).sin())
            .collect();
        let weights = [0.1, -0.2, 0.35, 0.75];
        let mut n = ParallelSpikingNeuron::new(4, 0.4);
        n.weights = weights.to_vec();
        let (hidden_trace, spikes) = oracle(&weights, 0.4, &drive);
        for (t, &current) in drive.iter().enumerate() {
            let spike = n.try_step(current).expect("finite configured drive");
            assert_eq!(n.hidden.to_bits(), hidden_trace[t].to_bits());
            assert_eq!(spike, spikes[t]);
        }
    }

    #[test]
    fn firing_never_clears_history() {
        let mut n = ParallelSpikingNeuron::new(4, 0.5);
        let mut fired = 0;
        for _ in 0..8 {
            fired += n.step(1.0);
        }
        assert!(
            fired >= 5,
            "constant supra-threshold drive must keep firing"
        );
        assert!(n.history.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn theta_is_right_continuous_at_threshold() {
        let mut n = ParallelSpikingNeuron::new(1, 1.0);
        assert_eq!(n.step(1.0), 1);
    }

    #[test]
    fn invalid_input_is_rejected_atomically() {
        let mut n = ParallelSpikingNeuron::new(4, 0.5);
        n.step(0.7);
        let before = (n.history.clone(), n.hidden);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(n.try_step(bad).is_err());
            assert_eq!(n.step(bad), 0);
        }
        assert_eq!((n.history.clone(), n.hidden), before);
    }

    #[test]
    fn overflowing_hidden_state_is_rejected_atomically() {
        let mut n = ParallelSpikingNeuron::new(2, 0.5);
        n.weights = vec![f64::MAX, f64::MAX];
        n.history = vec![f64::MAX, f64::MAX];
        assert!(n.try_step(f64::MAX).is_err());
        assert_eq!(n.history, vec![f64::MAX, f64::MAX]);
    }

    #[test]
    fn reset_clears_history_only() {
        let mut n = ParallelSpikingNeuron::new(4, 0.5);
        n.step(1.0);
        n.reset();
        assert!(n.history.iter().all(|&x| x == 0.0));
        assert_eq!(n.hidden, 0.0);
        assert_eq!(n.weights, vec![0.25; 4]);
    }
}
