// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety implementation of the published DPI circuit

pub type DpiCompleteTrace = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<u8>);

#[derive(Debug, Clone, PartialEq)]
pub struct DPINeuron {
    pub i_mem: f64,
    pub i_ahp: f64,
    pub refractory_time: f64,
    pub i_threshold: f64,
    pub i_reset: f64,
    pub i_rest: f64,
    pub i_tau: f64,
    pub i_g: f64,
    pub i_tau_ahp: f64,
    pub i_ga: f64,
    pub i_spike: f64,
    pub i_0: f64,
    pub kappa: f64,
    pub alpha: f64,
    pub tau: f64,
    pub tau_ahp: f64,
    pub refractory_period: f64,
    pub dt: f64,
}

impl DPINeuron {
    pub fn new() -> Self {
        Self {
            i_mem: 0.01,
            i_ahp: 0.01,
            refractory_time: 0.0,
            i_threshold: 1.0,
            i_reset: 0.01,
            i_rest: 0.1,
            i_tau: 1.0,
            i_g: 1.0,
            i_tau_ahp: 0.1,
            i_ga: 1.0,
            i_spike: 5.0,
            i_0: 0.01,
            kappa: 0.7,
            alpha: 10.0,
            tau: 20.0,
            tau_ahp: 100.0,
            refractory_period: 2.0,
            dt: 0.1,
        }
    }

    fn sigmoid(value: f64) -> f64 {
        if value >= 0.0 {
            1.0 / (1.0 + (-value).exp())
        } else {
            let exponential = value.exp();
            exponential / (1.0 + exponential)
        }
    }

    fn feedback_current(&self) -> f64 {
        let log_current = (self.i_0.ln() + self.kappa * self.i_mem.ln()) / (self.kappa + 1.0);
        log_current.exp() * Self::sigmoid(self.alpha * (self.i_mem - self.i_threshold))
    }

    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() || !validate_dpi_neuron(self) {
            return Err("DPI state, parameters, and current must be physically valid");
        }
        let total_input = self.i_rest + current;
        if !total_input.is_finite() || total_input < 0.0 {
            return Err("DPI total input current must be finite and non-negative");
        }

        let spike_active = self.refractory_time > 0.0;
        let spike_current = if spike_active { self.i_spike } else { 0.0 };
        let d_i_ahp = self.i_ahp / (self.tau_ahp * self.i_tau_ahp)
            * (spike_current / (1.0 + self.i_ahp / self.i_ga) - self.i_tau_ahp);
        let next_i_ahp = self.i_ahp + self.dt * d_i_ahp;

        let (next_i_mem, next_refractory, spiked) = if spike_active {
            (
                self.i_reset,
                (self.refractory_time - self.dt).max(0.0),
                false,
            )
        } else {
            let i_fb = self.feedback_current();
            let d_i_mem = self.i_mem / (self.tau * self.i_tau)
                * (total_input / (1.0 + self.i_mem / self.i_g) - self.i_tau + i_fb - self.i_ahp);
            let candidate = self.i_mem + self.dt * d_i_mem;
            if !candidate.is_finite() || candidate <= 0.0 {
                return Err("DPI membrane Euler candidate left the physical current domain");
            }
            if candidate >= self.i_threshold {
                (self.i_reset, self.refractory_period, true)
            } else {
                (candidate, 0.0, false)
            }
        };

        if !next_i_mem.is_finite()
            || !next_i_ahp.is_finite()
            || !next_refractory.is_finite()
            || next_i_mem <= 0.0
            || next_i_ahp < 0.0
            || next_refractory < 0.0
        {
            return Err("DPI Euler update left the physical current domain");
        }

        self.i_mem = next_i_mem;
        self.i_ahp = next_i_ahp;
        self.refractory_time = next_refractory;
        Ok(i32::from(spiked))
    }

    /// Return aligned complete traces and commit only after every step passes.
    pub fn simulate_complete(
        &mut self,
        n_steps: usize,
        current: f64,
    ) -> Result<DpiCompleteTrace, &'static str> {
        if !current.is_finite() || !validate_dpi_neuron(self) {
            return Err("DPI state, parameters, and current must be physically valid");
        }
        let total_input = self.i_rest + current;
        if !total_input.is_finite() || total_input < 0.0 {
            return Err("DPI total input current must be finite and non-negative");
        }
        let mut candidate = self.clone();
        let mut i_mem = Vec::with_capacity(n_steps);
        let mut i_ahp = Vec::with_capacity(n_steps);
        let mut refractory = Vec::with_capacity(n_steps);
        let mut events = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            let event = candidate.step(current)?;
            i_mem.push(candidate.i_mem);
            i_ahp.push(candidate.i_ahp);
            refractory.push(candidate.refractory_time);
            events.push(u8::try_from(event).map_err(|_| "DPI event must be binary")?);
        }
        *self = candidate;
        Ok((i_mem, i_ahp, refractory, events))
    }

    pub fn reset(&mut self) {
        self.i_mem = self.i_reset;
        self.i_ahp = self.i_0;
        self.refractory_time = 0.0;
    }
}

impl Default for DPINeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_dpi_neuron(state: &DPINeuron) -> bool {
    state.i_mem.is_finite()
        && state.i_mem > 0.0
        && state.i_ahp.is_finite()
        && state.i_ahp >= 0.0
        && state.refractory_time.is_finite()
        && state.refractory_time >= 0.0
        && state.i_threshold.is_finite()
        && state.i_threshold > 0.0
        && state.i_reset.is_finite()
        && state.i_reset > 0.0
        && state.i_reset < state.i_threshold
        && state.i_rest.is_finite()
        && state.i_rest >= 0.0
        && state.i_tau.is_finite()
        && state.i_tau > 0.0
        && state.i_g.is_finite()
        && state.i_g > 0.0
        && state.i_tau_ahp.is_finite()
        && state.i_tau_ahp > 0.0
        && state.i_ga.is_finite()
        && state.i_ga > 0.0
        && state.i_spike.is_finite()
        && state.i_spike > 0.0
        && state.i_0.is_finite()
        && state.i_0 > 0.0
        && state.kappa.is_finite()
        && state.kappa > 0.0
        && state.alpha.is_finite()
        && state.alpha > 0.0
        && state.tau.is_finite()
        && state.tau > 0.0
        && state.tau_ahp.is_finite()
        && state.tau_ahp > 0.0
        && state.refractory_period.is_finite()
        && state.refractory_period > 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.refractory_period >= state.dt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_state_is_valid() {
        assert!(validate_dpi_neuron(&DPINeuron::new()));
    }

    #[test]
    fn one_step_matches_coupled_euler_reference() {
        let mut state = DPINeuron::new();
        assert_eq!(state.step(5.0).unwrap(), 0);
        assert!((state.i_mem - 0.010201975272610835).abs() < 1.0e-17);
        assert!((state.i_ahp - 0.00999).abs() < 1.0e-17);
        assert_eq!(state.refractory_time, 0.0);
    }

    #[test]
    fn threshold_crossing_starts_refractory_pulse() {
        let mut state = DPINeuron {
            i_mem: 0.99,
            ..DPINeuron::new()
        };
        assert_eq!(state.step(10.0).unwrap(), 1);
        assert_eq!(state.i_mem, state.i_reset);
        assert_eq!(state.refractory_time, state.refractory_period);
    }

    #[test]
    fn refractory_pulse_holds_reset_and_drives_adaptation() {
        let mut state = DPINeuron {
            refractory_time: 2.0,
            ..DPINeuron::new()
        };
        assert_eq!(state.step(0.0).unwrap(), 0);
        assert_eq!(state.i_mem, state.i_reset);
        assert!(state.i_ahp > 0.01);
        assert_eq!(state.refractory_time, 1.9);
    }

    #[test]
    fn adaptation_decays_between_spike_pulses() {
        let mut state = DPINeuron::new();
        assert_eq!(state.step(0.0).unwrap(), 0);
        assert_eq!(state.i_ahp, 0.00999);
    }

    #[test]
    fn invalid_current_does_not_mutate_state() {
        let mut state = DPINeuron::new();
        let before = state.clone();
        assert!(state.step(f64::NAN).is_err());
        assert_eq!(state, before);
    }

    #[test]
    fn non_finite_candidate_does_not_mutate_state() {
        let mut state = DPINeuron {
            tau: f64::MIN_POSITIVE,
            ..DPINeuron::new()
        };
        let before = state.clone();
        assert!(state.step(f64::MAX).is_err());
        assert_eq!(state, before);
    }

    #[test]
    fn reset_preserves_parameters() {
        let mut state = DPINeuron {
            i_mem: 0.75,
            i_ahp: 0.4,
            refractory_time: 1.0,
            i_threshold: 1.3,
            i_reset: 0.2,
            i_0: 0.04,
            ..DPINeuron::new()
        };
        state.reset();
        assert_eq!(state.i_mem, 0.2);
        assert_eq!(state.i_ahp, 0.04);
        assert_eq!(state.refractory_time, 0.0);
        assert_eq!(state.i_threshold, 1.3);
        assert_eq!(state.i_reset, 0.2);
        assert_eq!(state.i_0, 0.04);
    }

    #[test]
    fn complete_packet_exposes_every_state_and_event() {
        let mut state = DPINeuron {
            i_mem: 0.37,
            i_ahp: 0.08,
            i_threshold: 1.3,
            i_reset: 0.2,
            i_rest: 0.15,
            i_tau: 0.9,
            i_g: 1.4,
            i_tau_ahp: 0.12,
            i_ga: 0.8,
            i_spike: 4.2,
            i_0: 0.02,
            kappa: 0.65,
            alpha: 8.0,
            tau: 7.0,
            tau_ahp: 45.0,
            refractory_period: 0.6,
            dt: 0.05,
            ..DPINeuron::new()
        };
        let (i_mem, i_ahp, refractory, events) = state.simulate_complete(400, 5.0).unwrap();
        assert_eq!(i_mem.len(), 400);
        assert_eq!(i_ahp.len(), 400);
        assert_eq!(refractory.len(), 400);
        assert_eq!(
            events
                .iter()
                .map(|event| usize::from(*event))
                .sum::<usize>(),
            4
        );
        assert_eq!(state.i_mem, i_mem[399]);
        assert_eq!(state.i_ahp, i_ahp[399]);
        assert_eq!(state.refractory_time, refractory[399]);
    }

    #[test]
    fn complete_packet_rejects_atomically() {
        let mut state = DPINeuron {
            tau: f64::MIN_POSITIVE,
            ..DPINeuron::new()
        };
        let before = state.clone();
        assert!(state.simulate_complete(2, f64::MAX).is_err());
        assert_eq!(state, before);
    }

    #[test]
    fn invalid_parameter_contract_is_rejected() {
        let mut state = DPINeuron {
            i_ga: 0.0,
            ..DPINeuron::new()
        };
        let before = state.clone();
        assert!(!validate_dpi_neuron(&state));
        assert!(state.step(1.0).is_err());
        assert_eq!(state, before);
    }

    #[test]
    fn sustained_drive_exhibits_spike_frequency_adaptation() {
        let mut state = DPINeuron::new();
        let spike_steps: Vec<usize> = (0..2_000)
            .filter(|_| state.step(5.0).unwrap() == 1)
            .collect();
        assert!(spike_steps.len() >= 5);
        let first_isi = spike_steps[1] - spike_steps[0];
        let last_isi = spike_steps[spike_steps.len() - 1] - spike_steps[spike_steps.len() - 2];
        assert!(last_isi > first_isi);
    }

    #[test]
    fn enrolled_event_vector_matches_python_reference() {
        let cases = [
            (-0.1, 0),
            (0.0, 0),
            (1.0, 0),
            (2.0, 0),
            (3.0, 1),
            (5.0, 3),
            (10.0, 6),
            (20.0, 11),
            (50.0, 21),
        ];
        for (current, expected) in cases {
            let mut state = DPINeuron::new();
            let spikes: i32 = (0..1_000).map(|_| state.step(current).unwrap()).sum();
            assert_eq!(spikes, expected, "current={current}");
        }
    }

    #[test]
    fn configured_contract_matches_python_golden() {
        let mut state = DPINeuron {
            i_mem: 0.37,
            i_ahp: 0.08,
            refractory_time: 0.0,
            i_threshold: 1.3,
            i_reset: 0.2,
            i_rest: 0.15,
            i_tau: 0.9,
            i_g: 1.4,
            i_tau_ahp: 0.12,
            i_ga: 0.8,
            i_spike: 4.2,
            i_0: 0.02,
            kappa: 0.65,
            alpha: 8.0,
            tau: 7.0,
            tau_ahp: 45.0,
            refractory_period: 0.6,
            dt: 0.05,
        };
        let spikes: i32 = (0..400).map(|_| state.step(5.0).unwrap()).sum();
        assert_eq!(spikes, 4);
        assert_eq!(state.i_mem, 0.2);
        assert!((state.i_ahp - 0.27412306389119817).abs() < 2.0e-15);
        assert_eq!(state.refractory_time, 0.0);
    }
}
