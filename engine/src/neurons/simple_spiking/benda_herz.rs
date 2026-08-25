// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — source-faithful Benda-Herz universal adaptation

//! Benda–Herz equations (8) and (45), specialized to the paper's square-root
//! onset curve and linear steady-state adaptation example.

/// Deterministic Benda–Herz universal rate-adaptation neuron.
#[derive(Clone, Debug)]
pub struct BendaHerzNeuron {
    /// Adaptation variable.
    pub a: f64,
    /// Phase of the deterministic spike generator in `[0, 1)`.
    pub phase: f64,
    /// Gain of `f0(x) = onset_gain * sqrt(max(x-rheobase, 0))`.
    pub onset_gain: f64,
    /// Onset-current offset.
    pub rheobase: f64,
    /// Slope in `A_inf(f) = adaptation_slope * f`.
    pub adaptation_slope: f64,
    /// Adaptation time constant in milliseconds.
    pub tau_a: f64,
    /// Sample interval in milliseconds.
    pub dt: f64,
}

impl Default for BendaHerzNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl BendaHerzNeuron {
    /// Construct the paper-example specialization at rest.
    #[must_use]
    pub fn new() -> Self {
        Self {
            a: 0.0,
            phase: 0.0,
            onset_gain: 60.0,
            rheobase: 0.0,
            adaptation_slope: 0.1,
            tau_a: 100.0,
            dt: 0.1,
        }
    }

    fn rate(&self, current: f64, a: f64) -> f64 {
        self.onset_gain * (current - a - self.rheobase).max(0.0).sqrt()
    }

    fn rhs(&self, current: f64, a: f64) -> (f64, f64) {
        let rate = self.rate(current, a);
        (
            (self.adaptation_slope * rate - a) / self.tau_a,
            rate / 1000.0,
        )
    }

    /// Advance one sample, returning `-1` if validation fails atomically.
    pub fn step(&mut self, current: f64) -> i32 {
        if !self.valid() || !current.is_finite() {
            return -1;
        }
        let (k1a, k1p) = self.rhs(current, self.a);
        let (k2a, k2p) = self.rhs(current, self.a + 0.5 * self.dt * k1a);
        let (k3a, k3p) = self.rhs(current, self.a + 0.5 * self.dt * k2a);
        let (k4a, k4p) = self.rhs(current, self.a + self.dt * k3a);
        let scale = self.dt / 6.0;
        let next_a = self.a + scale * (k1a + 2.0 * k2a + 2.0 * k3a + k4a);
        let next_phase = self.phase + scale * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
        if !next_a.is_finite()
            || next_a < 0.0
            || !next_phase.is_finite()
            || !(0.0..2.0).contains(&next_phase)
        {
            return -1;
        }
        self.a = next_a;
        if next_phase >= 1.0 {
            self.phase = 0.0;
            1
        } else {
            self.phase = next_phase;
            0
        }
    }

    /// Reset both source state variables.
    pub fn reset(&mut self) {
        self.a = 0.0;
        self.phase = 0.0;
    }

    /// Validate parameters and state.
    #[must_use]
    pub fn valid(&self) -> bool {
        self.a.is_finite()
            && self.a >= 0.0
            && self.phase.is_finite()
            && (0.0..1.0).contains(&self.phase)
            && self.onset_gain.is_finite()
            && self.onset_gain > 0.0
            && self.rheobase.is_finite()
            && self.adaptation_slope.is_finite()
            && self.adaptation_slope >= 0.0
            && self.tau_a.is_finite()
            && self.tau_a > 0.0
            && self.dt.is_finite()
            && self.dt > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_paper_example() {
        let neuron = BendaHerzNeuron::new();
        assert_eq!(neuron.onset_gain, 60.0);
        assert_eq!(neuron.adaptation_slope, 0.1);
        assert_eq!(neuron.a, 0.0);
        assert_eq!(neuron.phase, 0.0);
    }

    #[test]
    fn phase_spike_is_deterministic() {
        let mut left = BendaHerzNeuron::new();
        let mut right = BendaHerzNeuron::new();
        left.phase = 0.99;
        right.phase = 0.99;
        left.dt = 1.0;
        right.dt = 1.0;
        left.adaptation_slope = 0.0;
        right.adaptation_slope = 0.0;
        assert_eq!(left.step(1.0), 1);
        assert_eq!(right.step(1.0), 1);
        assert_eq!(left.phase, 0.0);
        assert_eq!(left.phase, right.phase);
        assert_eq!(left.a, right.a);
    }

    #[test]
    fn invalid_transition_is_atomic() {
        let mut neuron = BendaHerzNeuron::new();
        neuron.a = 0.5;
        neuron.phase = 0.5;
        neuron.onset_gain = 1.0e6;
        neuron.dt = 1.0;
        assert_eq!(neuron.step(1.0e6), -1);
        assert_eq!(neuron.a, 0.5);
        assert_eq!(neuron.phase, 0.5);
    }

    #[test]
    fn reset_clears_both_state_variables() {
        let mut neuron = BendaHerzNeuron::new();
        neuron.a = 1.0;
        neuron.phase = 0.25;
        neuron.reset();
        assert_eq!(neuron.a, 0.0);
        assert_eq!(neuron.phase, 0.0);
    }
}
