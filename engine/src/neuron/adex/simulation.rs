// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Atomic AdEx batch simulation

use super::AdExNeuron;

/// Aligned voltage, adaptation, and event traces from one AdEx batch.
pub type AdExSimulation = (Vec<f64>, Vec<f64>, Vec<u8>);

impl AdExNeuron {
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
}

#[cfg(test)]
mod tests {
    use super::AdExNeuron;

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
