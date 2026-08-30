// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Adaptive exponential integrate-and-fire neuron

/// Aligned voltage, adaptation, and event traces from one AdEx batch.
pub type AdExSimulation = (Vec<f64>, Vec<f64>, Vec<u8>);

/// Adaptive Exponential IF neuron. Brette & Gerstner 2005.
/// PyO3 wrapper: `pyo3_neurons::PyAdExNeuron`
#[derive(Clone, Debug)]
pub struct AdExNeuron {
    pub v: f64,
    pub w: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub v_rh: f64,
    pub delta_t: f64,
    pub tau: f64,
    pub tau_w: f64,
    pub a: f64,
    pub b: f64,
    pub c_m: f64,
    pub dt: f64,
}

impl Default for AdExNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl AdExNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            w: 0.0,
            v_rest: -65.0,
            v_reset: -68.0,
            v_threshold: -50.0,
            v_rh: -55.0,
            delta_t: 2.0,
            tau: 20.0,
            tau_w: 100.0,
            a: 0.5,
            b: 7.0,
            c_m: 200.0,
            dt: 0.1,
        }
    }

    /// Advance one maintained baseline-Euler step.
    ///
    /// This compatibility surface retains the historical zero-event result on
    /// invalid input. New batch and binding code uses [`Self::try_step`] so a
    /// rejected update cannot be mistaken for a valid quiet timestep.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Advance one checked baseline-Euler step without partial mutation.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !self.v.is_finite()
            || !self.w.is_finite()
            || !self.v_rest.is_finite()
            || !self.v_reset.is_finite()
            || !self.v_threshold.is_finite()
            || !self.v_rh.is_finite()
            || !self.delta_t.is_finite()
            || !self.tau.is_finite()
            || !self.tau_w.is_finite()
            || !self.a.is_finite()
            || !self.b.is_finite()
            || !self.c_m.is_finite()
            || !self.dt.is_finite()
            || !current.is_finite()
            || self.delta_t <= 0.0
            || self.tau <= 0.0
            || self.tau_w <= 0.0
            || self.c_m <= 0.0
            || self.dt <= 0.0
        {
            return Err("invalid AdEx state, parameters, timestep, or input");
        }

        let exp_arg = ((self.v - self.v_rh) / self.delta_t).clamp(-20.0, 20.0);
        let exp_term = self.delta_t * exp_arg.exp();
        let dv = ((-(self.v - self.v_rest) + exp_term) / self.tau + (-self.w + current) / self.c_m)
            * self.dt;
        let dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w * self.dt;
        let next_v = self.v + dv;
        let next_w = self.w + dw;
        if !exp_term.is_finite()
            || !dv.is_finite()
            || !dw.is_finite()
            || !next_v.is_finite()
            || !next_w.is_finite()
        {
            return Err("non-finite AdEx integrator candidate");
        }

        if next_v >= self.v_threshold {
            let spike_w = next_w + self.b;
            if !spike_w.is_finite() {
                return Err("non-finite AdEx spike-adaptation candidate");
            }
            self.v = self.v_reset;
            self.w = spike_w;
            Ok(1)
        } else {
            self.v = next_v;
            self.w = next_w;
            Ok(0)
        }
    }

    /// Return aligned voltage, adaptation, and event traces atomically.
    ///
    /// The receiver is committed only after every candidate step succeeds.
    pub fn simulate_complete(
        &mut self,
        n_steps: usize,
        current: f64,
    ) -> Result<AdExSimulation, &'static str> {
        let mut candidate = self.clone();
        let mut v_trace = Vec::with_capacity(n_steps);
        let mut w_trace = Vec::with_capacity(n_steps);
        let mut event_trace = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            let event = candidate.try_step(current)?;
            v_trace.push(candidate.v);
            w_trace.push(candidate.w);
            event_trace.push(u8::try_from(event).map_err(|_| "invalid AdEx event value")?);
        }
        self.v = candidate.v;
        self.w = candidate.w;
        Ok((v_trace, w_trace, event_trace))
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.w = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::AdExNeuron;

    #[test]
    fn strong_input_produces_spikes() {
        let mut neuron = AdExNeuron::new();
        let spikes: i32 = (0..2_000).map(|_| neuron.step(500.0)).sum();
        assert!(spikes > 0, "AdEx must fire with strong input");
    }

    #[test]
    fn adaptation_does_not_increase_late_rate() {
        let mut neuron = AdExNeuron::new();
        let first: i32 = (0..1_000).map(|_| neuron.step(400.0)).sum();
        let second: i32 = (0..1_000).map(|_| neuron.step(400.0)).sum();
        assert!(second <= first + 5, "first={first}, second={second}");
    }

    #[test]
    fn matches_python_golden_spike_counts() {
        for (current, expected) in [(0.0, 0), (200.0, 4), (500.0, 12)] {
            let mut neuron = AdExNeuron::new();
            let spikes: i32 = (0..1_000).map(|_| neuron.step(current)).sum();
            assert_eq!(spikes, expected, "current={current}");
        }
    }

    #[test]
    fn invalid_input_is_mutation_free() {
        let mut neuron = AdExNeuron::new();
        let before = (neuron.v, neuron.w);
        assert!(neuron.try_step(f64::INFINITY).is_err());
        assert_eq!((neuron.v, neuron.w), before);
    }

    #[test]
    fn nonfinite_candidate_is_mutation_free() {
        let mut neuron = AdExNeuron::new();
        neuron.dt = 1.0e308;
        let before = (neuron.v, neuron.w);
        assert!(neuron.try_step(1.0e308).is_err());
        assert_eq!((neuron.v, neuron.w), before);
    }

    #[test]
    fn no_input_remains_silent() {
        let mut neuron = AdExNeuron::new();
        let spikes: i32 = (0..1_000).map(|_| neuron.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn negative_current_remains_silent() {
        let mut neuron = AdExNeuron::new();
        let spikes: i32 = (0..500).map(|_| neuron.step(-100.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn reset_matches_fresh_neuron() {
        let mut neuron = AdExNeuron::new();
        for _ in 0..200 {
            neuron.step(500.0);
        }
        assert!(neuron.w > 0.0);
        neuron.reset();
        assert_eq!(neuron.v, neuron.v_rest);
        assert_eq!(neuron.w, 0.0);

        let mut fresh = AdExNeuron::new();
        let reset_spikes: i32 = (0..100).map(|_| neuron.step(500.0)).sum();
        let fresh_spikes: i32 = (0..100).map(|_| fresh.step(500.0)).sum();
        assert_eq!(reset_spikes, fresh_spikes);
    }

    #[test]
    fn sustained_high_input_keeps_state_finite() {
        let mut neuron = AdExNeuron::new();
        for _ in 0..5_000 {
            neuron.step(1_000.0);
        }
        assert!(neuron.v.is_finite());
        assert!(neuron.w.is_finite());
    }

    #[test]
    fn sustained_input_produces_many_spikes() {
        let mut neuron = AdExNeuron::new();
        let spikes: i32 = (0..10_000).map(|_| neuron.step(500.0)).sum();
        assert!(spikes > 100, "got {spikes}");
        assert!(neuron.v.is_finite());
    }

    #[test]
    fn ten_thousand_steps_complete_within_smoke_limit() {
        let mut neuron = AdExNeuron::new();
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            neuron.step(500.0);
        }
        assert!(start.elapsed().as_millis() < 50);
    }

    #[test]
    fn complete_batch_is_full_parameter_and_failure_atomic() {
        let mut neuron = AdExNeuron {
            v: -60.0,
            w: 3.0,
            v_rest: -64.0,
            v_reset: -69.0,
            v_threshold: -49.0,
            v_rh: -54.0,
            delta_t: 2.5,
            tau: 18.0,
            tau_w: 120.0,
            a: 0.7,
            b: 8.0,
            c_m: 180.0,
            dt: 0.2,
        };
        let (v_trace, w_trace, events) = neuron
            .simulate_complete(250, 410.0)
            .expect("finite configured AdEx trajectory");
        assert_eq!(
            (v_trace.len(), w_trace.len(), events.len()),
            (250, 250, 250)
        );
        assert_eq!(
            events
                .iter()
                .map(|event| usize::from(*event))
                .sum::<usize>(),
            5
        );
        assert_eq!((neuron.v, neuron.w), (v_trace[249], w_trace[249]));

        let mut rejected = AdExNeuron::new();
        rejected.dt = 1.0e308;
        let before = (rejected.v, rejected.w);
        assert!(rejected.simulate_complete(2, 1.0e308).is_err());
        assert_eq!((rejected.v, rejected.w), before);
    }
}
