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

        // 3. Update weight (Potentiation/Depression)
        // If post follows pre (potentiation)
        if post_spike {
            let dw = (self.trace_pre as i32 * params.a_plus as i32) >> self.fraction;
            let mut new_w = self.weight as i32 + dw;
            if new_w > params.w_max as i32 {
                new_w = params.w_max as i32;
            }
            self.weight = mask(new_w, self.data_width);
        }

        // If pre follows post (depression)
        if pre_spike {
            let dw = (self.trace_post as i32 * params.a_minus as i32) >> self.fraction;
            let mut new_w = self.weight as i32 + dw; // dw is negative usually
            if new_w < params.w_min as i32 {
                new_w = params.w_min as i32;
            }
            self.weight = mask(new_w, self.data_width);
        }
    }
}
