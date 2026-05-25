// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for perfect_integrator

#[derive(Debug, Clone)]
pub struct PerfectIntegratorNeuron {
    pub v: f64,
    pub c_m: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub dt: f64,
}

impl PerfectIntegratorNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0_f64,
            c_m: 1.0_f64,
            v_threshold: 1.0_f64,
            v_reset: 0.0_f64,
            dt: 0.1_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !i_ext.is_finite() || !validate_perfect_integrator(self) {
            return Err("perfect-integrator state/current must be finite and physically ordered");
        }

        let voltage_increment = i_ext / self.c_m * self.dt;
        let next_v = self.v + voltage_increment;
        if !voltage_increment.is_finite() || !next_v.is_finite() {
            return Err("perfect-integrator voltage increment became non-finite");
        }

        self.v = next_v;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            Ok(1)
        } else {
            Ok(0)
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_reset;
    }
}

impl Default for PerfectIntegratorNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_perfect_integrator(state: &PerfectIntegratorNeuron) -> bool {
    state.v.is_finite()
        && state.c_m.is_finite()
        && state.c_m > 0.0
        && state.v_threshold.is_finite()
        && state.v_reset.is_finite()
        && state.v_threshold > state.v_reset
        && state.v < state.v_threshold
        && state.dt.is_finite()
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_integrator_new() {
        let state = PerfectIntegratorNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_perfect_integrator(&state));
    }

    #[test]
    fn test_perfect_integrator_step() {
        let mut state = PerfectIntegratorNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_positive_current_spikes_and_resets() {
        let mut state = PerfectIntegratorNeuron::new();
        let mut spikes = 0;
        for _ in 0..100 {
            spikes += state.step(10.0).unwrap();
        }
        assert_eq!(spikes, 100);
        assert_eq!(state.v, state.v_reset);
    }

    #[test]
    fn test_invalid_current_does_not_mutate_state() {
        let mut state = PerfectIntegratorNeuron::new();
        state.v = 0.25;
        assert!(state.step(f64::NAN).is_err());
        assert_eq!(state.v, 0.25);
    }

    #[test]
    fn test_invalid_increment_does_not_mutate_state() {
        let mut state = PerfectIntegratorNeuron::new();
        state.v = 0.25;
        state.v_threshold = 1.0e308;
        state.c_m = 1.0e-308;
        assert!(state.step(1.0e308).is_err());
        assert_eq!(state.v, 0.25);
    }

    #[test]
    fn test_reset_preserves_parameters() {
        let mut state = PerfectIntegratorNeuron {
            v: 0.5,
            c_m: 2.0,
            v_threshold: 3.0,
            v_reset: -1.0,
            dt: 0.05,
        };
        state.reset();
        assert_eq!(state.v, -1.0);
        assert_eq!(state.c_m, 2.0);
        assert_eq!(state.v_threshold, 3.0);
        assert_eq!(state.dt, 0.05);
    }
}
