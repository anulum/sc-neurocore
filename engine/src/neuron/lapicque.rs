// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Lapicque integrate-and-fire neuron

/// Lapicque 1907 — classical RC integrate-and-fire.
#[derive(Clone, Debug)]
pub struct LapicqueNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau: f64,
    pub resistance: f64,
    pub dt: f64,
}

impl LapicqueNeuron {
    pub fn new(tau: f64, resistance: f64, threshold: f64, dt: f64) -> Self {
        Self {
            v: 0.0,
            v_rest: 0.0,
            v_reset: 0.0,
            v_threshold: threshold,
            tau,
            resistance,
            dt,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !self.v.is_finite()
            || !self.v_rest.is_finite()
            || !self.v_reset.is_finite()
            || !self.v_threshold.is_finite()
            || self.v_threshold <= self.v_rest
            || self.v_threshold <= self.v_reset
            || self.v >= self.v_threshold
            || !self.tau.is_finite()
            || self.tau <= 0.0
            || !self.resistance.is_finite()
            || self.resistance <= 0.0
            || !self.dt.is_finite()
            || self.dt <= 0.0
            || !current.is_finite()
        {
            return 0;
        }

        let v_inf = self.v_rest + self.resistance * current;
        let decay = (-self.dt / self.tau).exp();
        let next_v = v_inf + (self.v - v_inf) * decay;
        if !v_inf.is_finite() || !decay.is_finite() || !next_v.is_finite() {
            return 0;
        }
        self.v = next_v;

        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
    }
}

#[cfg(test)]
mod tests {
    use super::LapicqueNeuron;

    fn neuron() -> LapicqueNeuron {
        LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0)
    }

    #[test]
    fn sustained_input_produces_spikes() {
        let mut neuron = neuron();
        let spikes: i32 = (0..200).map(|_| neuron.step(5.0)).sum();
        assert!(spikes > 0);
    }

    #[test]
    fn reset_restores_resting_voltage() {
        let mut neuron = neuron();
        for _ in 0..50 {
            neuron.step(5.0);
        }
        neuron.reset();
        assert!(neuron.v.abs() < 1e-12);
    }

    #[test]
    fn exact_flow_matches_closed_form() {
        let mut neuron = LapicqueNeuron::new(20.0, 1.0, 1.0, 5.0);
        neuron.v = 0.25;
        let current = 0.5;
        let v0 = neuron.v;
        let v_inf = neuron.v_rest + neuron.resistance * current;
        let euler =
            v0 + (-(v0 - neuron.v_rest) + neuron.resistance * current) / neuron.tau * neuron.dt;
        let expected = v_inf + (v0 - v_inf) * (-neuron.dt / neuron.tau).exp();
        assert_eq!(neuron.step(current), 0);
        assert!((neuron.v - expected).abs() < 1e-15);
        assert!((neuron.v - euler).abs() > 1e-4);
    }

    #[test]
    fn zero_input_remains_silent() {
        let mut neuron = neuron();
        let spikes: i32 = (0..500).map(|_| neuron.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn negative_input_remains_silent() {
        let mut neuron = neuron();
        let spikes: i32 = (0..500).map(|_| neuron.step(-5.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn invalid_state_does_not_mutate() {
        let mut neuron = neuron();
        neuron.v = 0.25;
        neuron.tau = 0.0;
        assert_eq!(neuron.step(1.0), 0);
        assert_eq!(neuron.v, 0.25);
    }

    #[test]
    fn reset_matches_fresh_neuron() {
        let mut neuron = neuron();
        for _ in 0..100 {
            neuron.step(5.0);
        }
        neuron.reset();
        let mut fresh = self::neuron();
        let reset_spikes: i32 = (0..100).map(|_| neuron.step(5.0)).sum();
        let fresh_spikes: i32 = (0..100).map(|_| fresh.step(5.0)).sum();
        assert_eq!(reset_spikes, fresh_spikes);
    }

    #[test]
    fn high_input_keeps_voltage_finite() {
        let mut neuron = neuron();
        for _ in 0..5_000 {
            neuron.step(100.0);
        }
        assert!(neuron.v.is_finite());
    }

    #[test]
    fn higher_resistance_does_not_reduce_spike_count() {
        let mut low = LapicqueNeuron::new(20.0, 0.5, 1.0, 1.0);
        let mut high = LapicqueNeuron::new(20.0, 2.0, 1.0, 1.0);
        let low_spikes: i32 = (0..200).map(|_| low.step(1.0)).sum();
        let high_spikes: i32 = (0..200).map(|_| high.step(1.0)).sum();
        assert!(high_spikes >= low_spikes);
    }

    #[test]
    fn ten_thousand_steps_complete_within_smoke_limit() {
        let mut neuron = neuron();
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            neuron.step(5.0);
        }
        assert!(start.elapsed().as_millis() < 50);
    }

    #[test]
    fn sustained_pipeline_input_produces_many_spikes() {
        let mut neuron = neuron();
        let spikes: i32 = (0..10_000).map(|_| neuron.step(5.0)).sum();
        assert!(spikes > 100, "got {spikes}");
    }
}
