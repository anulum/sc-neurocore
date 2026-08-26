// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Standalone Rust safety mirror for GLMNeuron

#[derive(Debug, Clone)]
/// Standalone safety mirror of the point-process generalised linear model, matching the Python
/// reference recurrence and its atomic fail-closed contract.
pub struct GLMNeuron {
    pub mu: f64,
    pub dt_ms: f64,
    pub k: Vec<f64>,
    pub h: Vec<f64>,
    pub stim_buf: Vec<f64>,
    pub spike_buf: Vec<f64>,
}

impl GLMNeuron {
    /// Construct the reference filters and empty history buffers.
    pub fn new(n_k: usize, n_h: usize) -> Self {
        let k = (0..n_k).map(|i| (-(i as f64) / 3.0).exp() * 0.5).collect();
        let h = (0..n_h)
            .map(|t| -5.0 * (-(t as f64) / 2.0).exp() + 0.5 * (-(t as f64) / 10.0).exp())
            .collect();
        Self {
            mu: -3.0,
            dt_ms: 1.0,
            k,
            h,
            stim_buf: vec![0.0; n_k],
            spike_buf: vec![0.0; n_h],
        }
    }

    /// Advance one step consuming an explicit uniform sample in [0, 1).
    pub fn step(&mut self, stimulus: f64, uniform: f64) -> Result<i32, &'static str> {
        if !stimulus.is_finite() {
            return Err("stimulus must be finite");
        }
        if !uniform.is_finite() || !(0.0..1.0).contains(&uniform) {
            return Err("uniform must be finite and within [0, 1)");
        }
        if !validate_glm_neuron(self) {
            return Err("GLM state and parameters must satisfy the public bounds");
        }

        let nk = self.stim_buf.len();
        let nh = self.spike_buf.len();
        let mut stim_candidate = self.stim_buf.clone();
        for i in (1..nk).rev() {
            stim_candidate[i] = stim_candidate[i - 1];
        }
        if nk > 0 {
            stim_candidate[0] = stimulus;
        }
        let dot_k: f64 = self
            .k
            .iter()
            .zip(stim_candidate.iter())
            .map(|(a, b)| a * b)
            .sum();
        let dot_h: f64 = self
            .h
            .iter()
            .zip(self.spike_buf.iter())
            .map(|(a, b)| a * b)
            .sum();
        let log_rate = (dot_k + dot_h + self.mu).clamp(-20.0, 20.0);
        let p = log_rate.exp() * self.dt_ms / 1000.0;
        let spike = if uniform < p.min(1.0) { 1 } else { 0 };
        let mut spike_candidate = self.spike_buf.clone();
        for i in (1..nh).rev() {
            spike_candidate[i] = spike_candidate[i - 1];
        }
        if nh > 0 {
            spike_candidate[0] = spike as f64;
        }
        self.stim_buf = stim_candidate;
        self.spike_buf = spike_candidate;
        Ok(spike)
    }

    /// Restore dynamic state to the initial values, preserving parameters.
    pub fn reset(&mut self) {
        self.stim_buf.fill(0.0);
        self.spike_buf.fill(0.0);
    }
}

/// Return whether every state and configuration field is finite and
/// inside the public descriptor bounds.
pub fn validate_glm_neuron(state: &GLMNeuron) -> bool {
    state.mu.is_finite()
        && state.dt_ms.is_finite()
        && state.dt_ms > 0.0
        && state.dt_ms <= 1000.0
        && state.k.len() == state.stim_buf.len()
        && state.h.len() == state.spike_buf.len()
        && !state.k.is_empty()
        && !state.h.is_empty()
        && state.k.iter().all(|value| value.is_finite())
        && state.h.iter().all(|value| value.is_finite())
        && state.stim_buf.iter().all(|value| value.is_finite())
        && state.spike_buf.iter().all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_forced_spike_and_refractory_feedback() {
        let mut state = GLMNeuron::new(10, 20);
        let spike = state.step(20.0, 0.0).expect("finite drive");
        assert_eq!(spike, 1);
        assert_eq!(state.spike_buf[0], 1.0);
        assert_eq!(state.stim_buf[0], 20.0);
    }

    #[test]
    fn invalid_drive_and_uniform_are_atomic() {
        let mut state = GLMNeuron::new(10, 20);
        let before = state.clone();
        assert!(state.step(f64::NAN, 0.5).is_err());
        assert!(state.step(1.0, 1.0).is_err());
        assert!(state.step(1.0, f64::NAN).is_err());
        assert_eq!(state.stim_buf, before.stim_buf);
        assert_eq!(state.spike_buf, before.spike_buf);
    }

    #[test]
    fn invalid_configuration_is_atomic() {
        let mut state = GLMNeuron::new(10, 20);
        state.mu = f64::NAN;
        let before = state.clone();
        assert!(state.step(1.0, 0.5).is_err());
        assert_eq!(state.stim_buf, before.stim_buf);
    }

    #[test]
    fn reset_preserves_filters() {
        let mut state = GLMNeuron::new(10, 20);
        state.step(5.0, 0.0).expect("finite drive");
        let k_before = state.k.clone();
        state.reset();
        assert!(state.stim_buf.iter().all(|value| *value == 0.0));
        assert!(state.spike_buf.iter().all(|value| *value == 0.0));
        assert_eq!(state.k, k_before);
    }
}
