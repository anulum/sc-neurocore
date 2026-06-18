// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for srm0

#[derive(Debug, Clone)]
pub struct SRM0Neuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_eta: f64,
    pub eta_reset: f64,
    pub resistance: f64,
    pub dt: f64,
    pub eta: f64,
    pub t: f64,
    pub last_spike_time: f64,
}

impl SRM0Neuron {
    pub fn new() -> Self {
        Self {
            v: 0.0,
            v_rest: 0.0,
            v_threshold: 1.0,
            tau_m: 20.0,
            tau_eta: 50.0,
            eta_reset: 5.0,
            resistance: 1.0,
            dt: 1.0,
            eta: 0.0,
            t: 0.0,
            last_spike_time: -1000.0,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_srm0(self) || !i_ext.is_finite() {
            return -1;
        }
        let Some((next_v, next_eta)) = self.exact_candidate(i_ext) else {
            return -1;
        };
        let next_t = self.t + self.dt;
        if next_v >= self.v_threshold {
            self.v = self.v_rest;
            self.eta = -self.eta_reset;
            self.t = next_t;
            self.last_spike_time = next_t;
            return 1;
        }
        self.v = next_v;
        self.eta = next_eta;
        self.t = next_t;
        0
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.eta = 0.0;
        self.t = 0.0;
        self.last_spike_time = -1000.0;
    }

    pub fn get_state(&self) -> (f64, f64, f64) {
        (self.v, self.eta, self.t)
    }

    fn eta_coupling_integral(&self) -> f64 {
        let membrane_decay = (-self.dt / self.tau_m).exp();
        let eta_decay = (-self.dt / self.tau_eta).exp();
        let rate_delta = (1.0 / self.tau_m) - (1.0 / self.tau_eta);
        if rate_delta.abs() < 1.0e-14 {
            return self.dt * membrane_decay / self.tau_m;
        }
        (eta_decay - membrane_decay) / (self.tau_m * rate_delta)
    }

    fn exact_candidate(&self, i_ext: f64) -> Option<(f64, f64)> {
        let membrane_decay = (-self.dt / self.tau_m).exp();
        let eta_decay = (-self.dt / self.tau_eta).exp();
        let steady = self.v_rest + self.resistance * i_ext;
        let next_eta = self.eta * eta_decay;
        let next_v =
            steady + (self.v - steady) * membrane_decay + self.eta * self.eta_coupling_integral();
        if next_v.is_finite() && next_eta.is_finite() {
            Some((next_v, next_eta))
        } else {
            None
        }
    }
}

impl Default for SRM0Neuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_srm0(state: &SRM0Neuron) -> bool {
    state.v.is_finite()
        && state.v_rest.is_finite()
        && state.v_threshold.is_finite()
        && state.tau_m.is_finite()
        && state.tau_m > 0.0
        && state.tau_eta.is_finite()
        && state.tau_eta > 0.0
        && state.eta_reset.is_finite()
        && state.eta_reset >= 0.0
        && state.resistance.is_finite()
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.eta.is_finite()
        && state.t.is_finite()
        && state.last_spike_time.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_step(state: &SRM0Neuron, current: f64) -> (f64, f64) {
        let membrane_decay = (-state.dt / state.tau_m).exp();
        let eta_decay = (-state.dt / state.tau_eta).exp();
        let rate_delta = (1.0 / state.tau_m) - (1.0 / state.tau_eta);
        let eta_coupling = if rate_delta.abs() < 1.0e-14 {
            state.dt * membrane_decay / state.tau_m
        } else {
            (eta_decay - membrane_decay) / (state.tau_m * rate_delta)
        };
        let steady = state.v_rest + state.resistance * current;
        (
            steady + (state.v - steady) * membrane_decay + state.eta * eta_coupling,
            state.eta * eta_decay,
        )
    }

    #[test]
    fn test_srm0_new() {
        let state = SRM0Neuron::new();
        assert!(validate_srm0(&state));
    }

    #[test]
    fn test_srm0_step_matches_exact_flow() {
        let mut state = SRM0Neuron {
            eta: -2.0,
            ..Default::default()
        };
        let (want_v, want_eta) = reference_step(&state, 0.5);
        assert_eq!(state.step(0.5), 0);
        assert!((state.v - want_v).abs() < 1.0e-12);
        assert!((state.eta - want_eta).abs() < 1.0e-12);
    }

    #[test]
    fn test_srm0_rejects_invalid_without_mutation() {
        let mut state = SRM0Neuron::new();
        let before = state.get_state();
        assert_eq!(state.step(f64::NAN), -1);
        assert_eq!(state.get_state(), before);
    }
}
