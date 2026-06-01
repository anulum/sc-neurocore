// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for non_resetting_lif

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct NonResettingLIFNeuron {
    pub v: f64,
    pub theta: f64,
    pub v_rest: f64,
    pub theta_rest: f64,
    pub delta_theta: f64,
    pub tau_m: f64,
    pub tau_theta: f64,
    pub r_m: f64,
    pub dt: f64,
}

impl NonResettingLIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0_f64,
            theta: -50.0_f64,
            v_rest: -65.0_f64,
            theta_rest: -50.0_f64,
            delta_theta: 5.0_f64,
            tau_m: 10.0_f64,
            tau_theta: 50.0_f64,
            r_m: 1.0_f64,
            dt: 0.1_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !i_ext.is_finite() || !validate_non_resetting_lif(self) {
            return Err("non-resetting-lif state/current must be finite and physically valid");
        }
        let membrane_steady_state = self.v_rest + self.r_m * i_ext;
        if !membrane_steady_state.is_finite() {
            return Err("non-resetting-lif exact relaxation update became non-finite");
        }
        let next_v = exact_relaxation(self.v, membrane_steady_state, self.dt, self.tau_m);
        if !next_v.is_finite() {
            return Err("non-resetting-lif exact relaxation update became non-finite");
        }
        let mut next_theta = exact_relaxation(self.theta, self.theta_rest, self.dt, self.tau_theta);
        if !next_theta.is_finite() {
            return Err("non-resetting-lif exact relaxation update became non-finite");
        }
        let spike = if next_v >= next_theta {
            next_theta += self.delta_theta;
            if !next_theta.is_finite() {
                return Err("non-resetting-lif exact relaxation update became non-finite");
            }
            1
        } else {
            0
        };
        self.v = next_v;
        self.theta = next_theta;
        Ok(spike)
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.theta = self.theta_rest;
    }
}

fn exact_relaxation(state: f64, steady_state: f64, dt: f64, tau: f64) -> f64 {
    let decay = (-dt / tau).exp();
    decay * state + (1.0 - decay) * steady_state
}

pub fn validate_non_resetting_lif(state: &NonResettingLIFNeuron) -> bool {
    state.v.is_finite()
        && state.theta.is_finite()
        && state.v_rest.is_finite()
        && state.theta_rest.is_finite()
        && state.delta_theta.is_finite()
        && state.delta_theta >= 0.0
        && state.tau_m.is_finite()
        && state.tau_m > 0.0
        && state.tau_theta.is_finite()
        && state.tau_theta > 0.0
        && state.r_m.is_finite()
        && state.r_m >= 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_reference(state: f64, steady_state: f64, dt: f64, tau: f64) -> f64 {
        let decay = (-dt / tau).exp();
        decay * state + (1.0 - decay) * steady_state
    }

    #[test]
    fn test_non_resetting_lif_new() {
        let state = NonResettingLIFNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_non_resetting_lif(&state));
    }

    #[test]
    fn test_non_resetting_lif_step() {
        let mut state = NonResettingLIFNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_positive_current_advances_membrane() {
        let mut state = NonResettingLIFNeuron::new();
        state.step(20.0).unwrap();
        assert!(state.v > -65.0);
    }

    #[test]
    fn test_exact_relaxation_matches_reference() {
        let mut state = NonResettingLIFNeuron::new();
        state.v = -60.0;
        state.theta = -40.0;
        state.dt = 0.5;
        let expected_v = exact_reference(
            state.v,
            state.v_rest + state.r_m * 4.0,
            state.dt,
            state.tau_m,
        );
        let expected_theta =
            exact_reference(state.theta, state.theta_rest, state.dt, state.tau_theta);
        let spike = state.step(4.0).unwrap();
        assert_eq!(spike, 0);
        assert!((state.v - expected_v).abs() < 1.0e-12);
        assert!((state.theta - expected_theta).abs() < 1.0e-12);
    }

    #[test]
    fn test_large_timestep_relaxation_remains_bounded() {
        let mut state = NonResettingLIFNeuron::new();
        state.v = 1000.0;
        state.theta = 2000.0;
        state.dt = 100.0;
        let spike = state.step(0.0).unwrap();
        assert_eq!(spike, 0);
        assert!(state.v >= state.v_rest && state.v <= 1000.0);
        assert!(state.theta >= state.theta_rest && state.theta <= 2000.0);
    }

    #[test]
    fn test_invalid_update_does_not_mutate_state() {
        let mut state = NonResettingLIFNeuron::new();
        state.v = -60.0;
        state.theta = -45.0;
        state.r_m = 10.0;
        let before = (state.v, state.theta);
        assert!(state.step(1.0e308).is_err());
        assert_eq!((state.v, state.theta), before);
    }
}
