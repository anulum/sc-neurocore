// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for spike_response

#[derive(Debug, Clone)]
pub struct SpikeResponseNeuron {
    pub v: f64,
    pub v_threshold: f64,
    pub tau_eta: f64,
    pub tau_kappa: f64,
    pub eta_reset: f64,
    pub time_since_spike: f64,
    pub dt: f64,
}

impl SpikeResponseNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0_f64,
            v_threshold: 1.0_f64,
            tau_eta: 10.0_f64,
            tau_kappa: 5.0_f64,
            eta_reset: -5.0_f64,
            time_since_spike: 1000.0_f64,
            dt: 1.0_f64,
        }
    }

    pub fn step(&mut self, weighted_input: f64) -> i32 {
        if !validate_spike_response(self) || !weighted_input.is_finite() {
            return -1;
        }
        let eta = if self.time_since_spike < 100.0 {
            self.eta_reset * (-self.time_since_spike / self.tau_eta).exp()
        } else {
            0.0
        };
        let kappa = weighted_input * (1.0 - (-self.dt / self.tau_kappa).exp());
        let next_v = eta + kappa;
        if !next_v.is_finite() {
            return -1;
        }
        self.v = next_v;
        self.time_since_spike += self.dt;
        if self.v >= self.v_threshold {
            self.time_since_spike = 0.0;
            self.v = 0.0;
            return 1;
        }
        0
    }

    pub fn reset(&mut self) {
        self.v = 0.0_f64;
        self.time_since_spike = 1000.0_f64;
    }
}

pub fn validate_spike_response(state: &SpikeResponseNeuron) -> bool {
    state.v.is_finite()
        && state.v_threshold.is_finite()
        && state.tau_eta.is_finite()
        && state.tau_eta > 0.0
        && state.tau_kappa.is_finite()
        && state.tau_kappa > 0.0
        && state.eta_reset.is_finite()
        && state.time_since_spike.is_finite()
        && state.time_since_spike >= 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_response_new() {
        let state = SpikeResponseNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_spike_response(&state));
    }

    #[test]
    fn test_spike_response_step() {
        let mut state = SpikeResponseNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_spike_response_kernel_step() {
        let mut state = SpikeResponseNeuron::new();
        let expected = 10.0 * (1.0 - (-state.dt / state.tau_kappa).exp());
        assert_eq!(state.step(10.0), 1);
        assert_eq!(state.v, 0.0);
        assert_eq!(state.time_since_spike, 0.0);

        assert_eq!(state.step(0.0), 0);
        assert!((state.v - state.eta_reset).abs() < 1.0e-12);
        assert!(expected > state.v_threshold);
    }

    #[test]
    fn test_spike_response_rejects_invalid_input_without_mutation() {
        let mut state = SpikeResponseNeuron::new();
        let original = (state.v, state.time_since_spike);
        assert_eq!(state.step(f64::NAN), -1);
        assert_eq!((state.v, state.time_since_spike), original);
    }
}
