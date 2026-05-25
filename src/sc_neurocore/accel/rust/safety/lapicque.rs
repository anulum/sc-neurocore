// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for lapicque

#[derive(Debug, Clone)]
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
    pub fn new() -> Self {
        Self {
            v: 0.0_f64,
            v_rest: 0.0_f64,
            v_reset: 0.0_f64,
            v_threshold: 1.0_f64,
            tau: 20.0_f64,
            resistance: 1.0_f64,
            dt: 1.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !i_ext.is_finite() {
            return Err("lapicque input current must be finite");
        }
        if !validate_lapicque(self) {
            return Err("lapicque state must satisfy finite positive-RC threshold contract");
        }

        let dv = (-(self.v - self.v_rest) + self.resistance * i_ext) / self.tau * self.dt;
        let next_v = self.v + dv;
        if !dv.is_finite() || !next_v.is_finite() {
            return Err("lapicque voltage increment must remain finite");
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
        self.v = self.v_rest;
    }
}

impl Default for LapicqueNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_lapicque(state: &LapicqueNeuron) -> bool {
    state.v.is_finite()
        && state.v_rest.is_finite()
        && state.v_reset.is_finite()
        && state.v_threshold.is_finite()
        && state.v_threshold > state.v_rest
        && state.v_threshold > state.v_reset
        && state.v < state.v_threshold
        && state.tau.is_finite()
        && state.tau > 0.0
        && state.resistance.is_finite()
        && state.resistance > 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lapicque_new() {
        let state = LapicqueNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_lapicque(&state));
    }

    #[test]
    fn test_lapicque_step() {
        let mut state = LapicqueNeuron::new();
        let spike = state.step(10.0).expect("valid step must succeed");
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_positive_current_spikes_and_resets() {
        let mut state = LapicqueNeuron::new();
        let mut spikes = 0;
        for _ in 0..5_000 {
            spikes += state.step(20.0).expect("valid step must succeed");
        }
        assert!(spikes >= 100);
        assert!(state.v < state.v_threshold);
    }

    #[test]
    fn test_invalid_current_does_not_mutate_state() {
        let mut state = LapicqueNeuron::new();
        state.v = 0.25;
        assert!(state.step(f64::NAN).is_err());
        assert_eq!(state.v, 0.25);
    }

    #[test]
    fn test_invalid_runtime_state_does_not_mutate_state() {
        let mut state = LapicqueNeuron::new();
        state.v = 0.25;
        state.tau = 0.0;
        assert!(state.step(1.0).is_err());
        assert_eq!(state.v, 0.25);
    }

    #[test]
    fn test_invalid_increment_does_not_mutate_state() {
        let mut state = LapicqueNeuron::new();
        state.v = 0.25;
        state.v_threshold = 1.0e308;
        state.tau = 1.0e-308;
        assert!(state.step(1.0e308).is_err());
        assert_eq!(state.v, 0.25);
    }

    #[test]
    fn test_reset_preserves_parameters() {
        let mut state = LapicqueNeuron {
            v: 0.5,
            v_rest: -0.25,
            v_reset: -0.5,
            v_threshold: 2.0,
            tau: 10.0,
            resistance: 2.0,
            dt: 0.25,
        };
        state.reset();
        assert_eq!(state.v, -0.25);
        assert_eq!(state.v_reset, -0.5);
        assert_eq!(state.v_threshold, 2.0);
        assert_eq!(state.tau, 10.0);
        assert_eq!(state.resistance, 2.0);
        assert_eq!(state.dt, 0.25);
    }
}
