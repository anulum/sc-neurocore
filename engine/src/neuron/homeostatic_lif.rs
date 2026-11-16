// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Homeostatic LIF neuron

#![deny(missing_docs)]

/// Homeostatic LIF neuron with adaptive threshold.
///
/// Threshold adapts via EMA of spike rate toward a target setpoint.
/// Turrigiano, Cold Spring Harb Perspect Biol 4:a005736, 2012.
#[derive(Clone, Debug)]
pub struct HomeostaticLif {
    /// Membrane potential (dimensionless).
    pub v: f64,
    /// Adaptive threshold, kept within `[0.1, 10]`.
    pub v_threshold: f64,
    /// Resting potential (dimensionless).
    pub v_rest: f64,
    /// Potential after a spike (dimensionless).
    pub v_reset: f64,
    /// Exponential moving average of the spike output, in spikes per step.
    pub rate_trace: f64,
    /// Target value of `rate_trace` (spikes per step).
    pub target_rate: f64,
    /// Threshold change per unit of rate error and step.
    pub adaptation_rate: f64,
    /// Per-step decay factor of `rate_trace`.
    pub trace_decay: f64,
    initial_threshold: f64,
}

impl HomeostaticLif {
    /// Create a neuron at rest with threshold 1 and the given homeostasis parameters.
    pub fn new(target_rate: f64, adaptation_rate: f64, trace_decay: f64) -> Self {
        Self {
            v: 0.0,
            v_threshold: 1.0,
            v_rest: 0.0,
            v_reset: 0.0,
            rate_trace: 0.0,
            target_rate,
            adaptation_rate,
            trace_decay,
            initial_threshold: 1.0,
        }
    }

    /// Create a neuron with target rate 0.1, adaptation rate 0.01 and trace decay 0.95.
    pub fn with_defaults() -> Self {
        Self::new(0.1, 0.01, 0.95)
    }

    /// LIF step with threshold adaptation. Returns 1 on spike.
    ///
    /// `v` gains `(-(v - v_rest) + current) / 20`: a fixed 20-step time constant, with no
    /// `dt`. Reaching the threshold sets `v` to `v_reset`. The rate trace then moves towards
    /// the spike output, and the threshold moves by `adaptation_rate * (rate_trace -
    /// target_rate)`, clamped to `[0.1, 10]`.
    pub fn step(&mut self, current: f64) -> i32 {
        let tau = 20.0;
        self.v += (-(self.v - self.v_rest) + current) / tau;

        let spike = if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        };

        self.rate_trace =
            self.rate_trace * self.trace_decay + spike as f64 * (1.0 - self.trace_decay);
        let error = self.rate_trace - self.target_rate;
        self.v_threshold += self.adaptation_rate * error;
        self.v_threshold = self.v_threshold.clamp(0.1, self.initial_threshold * 10.0);

        spike
    }

    /// Return `v` to `v_rest`, clear the rate trace and restore the threshold to 1; the
    /// homeostasis parameters are kept.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.rate_trace = 0.0;
        self.v_threshold = self.initial_threshold;
    }
}

#[cfg(test)]
#[path = "homeostatic_lif_tests.rs"]
mod tests;
