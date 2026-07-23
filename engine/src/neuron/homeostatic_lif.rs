// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Homeostatic LIF neuron

/// Homeostatic LIF neuron with adaptive threshold.
///
/// Threshold adapts via EMA of spike rate toward a target setpoint.
/// Turrigiano, Cold Spring Harb Perspect Biol 4:a005736, 2012.
#[derive(Clone, Debug)]
pub struct HomeostaticLif {
    pub v: f64,
    pub v_threshold: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub rate_trace: f64,
    pub target_rate: f64,
    pub adaptation_rate: f64,
    pub trace_decay: f64,
    initial_threshold: f64,
}

impl HomeostaticLif {
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

    pub fn with_defaults() -> Self {
        Self::new(0.1, 0.01, 0.95)
    }

    /// LIF step with threshold adaptation. Returns 1 on spike.
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

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.rate_trace = 0.0;
        self.v_threshold = self.initial_threshold;
    }
}

#[cfg(test)]
mod tests {
    use super::HomeostaticLif;

    #[test]
    fn strong_input_produces_spikes() {
        let mut neuron = HomeostaticLif::with_defaults();
        let spikes: i32 = (0..200).map(|_| neuron.step(25.0)).sum();
        assert!(spikes > 0, "must fire with strong input");
    }

    #[test]
    fn repeated_spiking_adapts_threshold() {
        let mut neuron = HomeostaticLif::with_defaults();
        let initial = neuron.v_threshold;
        for _ in 0..500 {
            neuron.step(25.0);
        }
        assert!(
            (neuron.v_threshold - initial).abs() > 1e-6,
            "threshold must adapt"
        );
    }

    #[test]
    fn zero_input_remains_silent() {
        let mut neuron = HomeostaticLif::with_defaults();
        let spikes: i32 = (0..100).map(|_| neuron.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn adaptive_threshold_remains_bounded() {
        let mut neuron = HomeostaticLif::with_defaults();
        for _ in 0..10_000 {
            neuron.step(50.0);
        }
        assert!((0.1..=10.0).contains(&neuron.v_threshold));
    }
}
