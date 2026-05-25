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
        let membrane_update = (-(self.v - self.v_rest) + self.r_m * i_ext) / self.tau_m * self.dt;
        let next_v = self.v + membrane_update;
        if !membrane_update.is_finite() || !next_v.is_finite() {
            return Err("non-resetting-lif membrane update became non-finite");
        }
        let threshold_update = (-(self.theta - self.theta_rest)) / self.tau_theta * self.dt;
        let mut next_theta = self.theta + threshold_update;
        if !threshold_update.is_finite() || !next_theta.is_finite() {
            return Err("non-resetting-lif threshold update became non-finite");
        }
        let spike = if next_v >= next_theta {
            next_theta += self.delta_theta;
            if !next_theta.is_finite() {
                return Err("non-resetting-lif threshold update became non-finite");
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
    fn test_invalid_update_does_not_mutate_state() {
        let mut state = NonResettingLIFNeuron::new();
        state.v = -60.0;
        state.theta = -45.0;
        state.tau_m = 1.0e-308;
        let before = (state.v, state.theta);
        assert!(state.step(1.0e308).is_err());
        assert_eq!((state.v, state.theta), before);
    }
}
