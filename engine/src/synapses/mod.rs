// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Stochastic STDP Synapse
//!
//! Implements Spike-Timing-Dependent Plasticity (STDP) for stochastic bitstreams.
//! Optimized for on-chip adaptation.

use crate::neuron::mask;

/// Parameters for the STDP rule.
#[derive(Clone, Copy, Debug)]
pub struct StdpParams {
    pub a_plus: i16,
    pub a_minus: i16,
    pub decay: i16,
    pub w_min: i16,
    pub w_max: i16,
}

/// A synapse with STDP learning capability.
#[derive(Clone, Debug)]
pub struct StdpSynapse {
    /// Current weight value (fixed-point).
    pub weight: i16,
    /// Pre-synaptic trace.
    pub trace_pre: i16,
    /// Post-synaptic trace.
    pub trace_post: i16,
    /// Data width.
    pub data_width: u32,
    /// Fraction bits.
    pub fraction: u32,
}

impl StdpSynapse {
    pub fn new(initial_weight: i16, data_width: u32, fraction: u32) -> Self {
        Self {
            weight: initial_weight,
            trace_pre: 0,
            trace_post: 0,
            data_width,
            fraction,
        }
    }

    /// Update traces and weight based on pre and post spikes.
    ///
    /// This is a simplified, hardware-friendly STDP rule:
    ///   - On pre-spike: trace_pre += A_plus; weight += trace_post * rate
    ///   - On post-spike: trace_post += A_minus; weight += trace_pre * rate
    ///   - Traces decay over time.
    pub fn step(&mut self, pre_spike: bool, post_spike: bool, params: &StdpParams) {
        // 1. Decay traces
        self.trace_pre = mask(
            (self.trace_pre as i32 * params.decay as i32) >> self.fraction,
            self.data_width,
        );
        self.trace_post = mask(
            (self.trace_post as i32 * params.decay as i32) >> self.fraction,
            self.data_width,
        );

        // 2. Update traces
        if pre_spike {
            self.trace_pre = mask(
                self.trace_pre as i32 + params.a_plus as i32,
                self.data_width,
            );
        }
        if post_spike {
            self.trace_post = mask(
                self.trace_post as i32 + params.a_minus as i32,
                self.data_width,
            );
        }

        // 3. Update weight — mutually exclusive per timestep.
        // Simultaneous spikes → LTP (pre-before-post convention).
        if post_spike {
            let dw = (self.trace_pre as i32 * params.a_plus.abs() as i32) >> self.fraction;
            let new_w = (self.weight as i32 + dw).min(params.w_max as i32);
            self.weight = mask(new_w, self.data_width);
        } else if pre_spike {
            let dw = (self.trace_post as i32 * params.a_minus.abs() as i32) >> self.fraction;
            let new_w = (self.weight as i32 - dw).max(params.w_min as i32);
            self.weight = mask(new_w, self.data_width);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> StdpParams {
        StdpParams {
            a_plus: 64,  // 0.25 in Q8.8
            a_minus: 48, // 0.1875 in Q8.8
            decay: 230,  // ~0.90 in Q8.8
            w_min: 0,
            w_max: 255,
        }
    }

    #[test]
    fn potentiation_increases_weight() {
        let mut syn = StdpSynapse::new(128, 16, 8);
        let params = default_params();
        // Multiple pre spikes to build trace
        for _ in 0..5 {
            syn.step(true, false, &params);
        }
        let w_before = syn.weight;
        // Post spike triggers LTP via accumulated pre-trace
        syn.step(false, true, &params);
        assert!(syn.weight > w_before, "LTP must increase weight");
    }

    #[test]
    fn depression_decreases_weight() {
        let mut syn = StdpSynapse::new(128, 16, 8);
        let params = default_params();
        // Multiple post spikes to build trace
        for _ in 0..5 {
            syn.step(false, true, &params);
        }
        let w_before = syn.weight;
        // Pre spike triggers LTD via accumulated post-trace
        syn.step(true, false, &params);
        assert!(syn.weight < w_before, "LTD must decrease weight");
    }

    #[test]
    fn weight_stays_in_bounds() {
        let mut syn = StdpSynapse::new(0, 16, 8);
        let params = default_params();
        for _ in 0..200 {
            syn.step(true, false, &params);
        }
        assert!(syn.weight >= params.w_min, "weight below w_min");
        assert!(syn.weight <= params.w_max, "weight above w_max");

        let mut syn2 = StdpSynapse::new(255, 16, 8);
        for _ in 0..200 {
            syn2.step(false, true, &params);
        }
        assert!(syn2.weight >= params.w_min);
        assert!(syn2.weight <= params.w_max);
    }
}
