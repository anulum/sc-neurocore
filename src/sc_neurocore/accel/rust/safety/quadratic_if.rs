// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for quadratic_if

#[derive(Debug, Clone)]
pub struct QuadraticIFNeuron {
    pub v: f64,
    pub v_reset: f64,
    pub v_peak: f64,
    pub dt: f64,
}

impl QuadraticIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -1.0_f64,
            v_reset: -1.0_f64,
            v_peak: 1.0_f64,
            dt: 0.01_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !i_ext.is_finite() || !validate_quadratic_if(self) {
            return 0;
        }

        let derivative = self.v * self.v + i_ext;
        let increment = derivative * self.dt;
        let next_v = self.v + increment;
        if !increment.is_finite() || !next_v.is_finite() {
            return 0;
        }

        self.v = next_v;
        if self.v >= self.v_peak {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_reset;
    }
}

impl Default for QuadraticIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_quadratic_if(state: &QuadraticIFNeuron) -> bool {
    state.v.is_finite()
        && state.v_reset.is_finite()
        && state.v_peak.is_finite()
        && state.dt.is_finite()
        && state.v < state.v_peak
        && state.v_reset < state.v_peak
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadratic_if_new() {
        let state = QuadraticIFNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_quadratic_if(&state));
    }

    #[test]
    fn test_quadratic_if_step() {
        let mut state = QuadraticIFNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_positive_current_spikes_and_resets() {
        let mut state = QuadraticIFNeuron::new();
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += state.step(1.0);
        }
        assert!(spikes >= 100);
        assert!(state.v < state.v_peak);
    }

    #[test]
    fn test_invalid_current_does_not_mutate_state() {
        let mut state = QuadraticIFNeuron::new();
        state.v = -0.25;
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!(state.v, -0.25);
    }

    #[test]
    fn test_invalid_euler_increment_does_not_mutate_state() {
        let mut state = QuadraticIFNeuron::new();
        state.v = -1.0e200;
        assert_eq!(state.step(0.0), 0);
        assert_eq!(state.v, -1.0e200);
    }

    #[test]
    fn test_reset_preserves_parameters() {
        let mut state = QuadraticIFNeuron {
            v: 0.5,
            v_reset: -2.0,
            v_peak: 2.0,
            dt: 0.005,
        };
        state.reset();
        assert_eq!(state.v, -2.0);
        assert_eq!(state.v_peak, 2.0);
        assert_eq!(state.dt, 0.005);
    }
}
