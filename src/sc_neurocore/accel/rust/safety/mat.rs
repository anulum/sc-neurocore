// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for mat

const MAT_V_MIN: f64 = -200.0_f64;
const MAT_V_MAX: f64 = 100.0_f64;
const MAT_THETA_MAX: f64 = 1.0e9_f64;

#[derive(Debug, Clone)]
pub struct MATNeuron {
    pub v: f64,
    pub theta1: f64,
    pub theta2: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold_base: f64,
    pub tau_m: f64,
    pub tau_1: f64,
    pub tau_2: f64,
    pub h1: f64,
    pub h2: f64,
    pub resistance: f64,
    pub dt: f64,
}

impl MATNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0_f64,
            theta1: 0.0_f64,
            theta2: 0.0_f64,
            v_rest: -70.0_f64,
            v_reset: -70.0_f64,
            v_threshold_base: -50.0_f64,
            tau_m: 10.0_f64,
            tau_1: 10.0_f64,
            tau_2: 200.0_f64,
            h1: 5.0_f64,
            h2: 3.0_f64,
            resistance: 1.0_f64,
            dt: 1.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !i_ext.is_finite() || !self.validate() {
            return -1;
        }
        let (v_candidate, theta1_candidate, theta2_candidate) =
            self.rk4_candidate(self.v, self.theta1, self.theta2, i_ext);
        if !mat_candidate_valid(v_candidate, theta1_candidate, theta2_candidate) {
            return -1;
        }
        let threshold = self.v_threshold_base + theta1_candidate + theta2_candidate;
        if v_candidate >= threshold {
            let theta1_after_spike = theta1_candidate + self.h1;
            let theta2_after_spike = theta2_candidate + self.h2;
            if !(theta1_after_spike.is_finite()
                && theta2_after_spike.is_finite()
                && theta1_after_spike <= MAT_THETA_MAX
                && theta2_after_spike <= MAT_THETA_MAX)
            {
                return -1;
            }
            self.v = self.v_reset;
            self.theta1 = theta1_after_spike;
            self.theta2 = theta2_after_spike;
            return 1;
        }
        self.v = v_candidate;
        self.theta1 = theta1_candidate;
        self.theta2 = theta2_candidate;
        0
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.theta1 = 0.0_f64;
        self.theta2 = 0.0_f64;
    }

    fn validate(&self) -> bool {
        self.v.is_finite()
            && self.theta1.is_finite()
            && self.theta2.is_finite()
            && self.v_rest.is_finite()
            && self.v_reset.is_finite()
            && self.v_threshold_base.is_finite()
            && self.tau_m.is_finite()
            && self.tau_1.is_finite()
            && self.tau_2.is_finite()
            && self.h1.is_finite()
            && self.h2.is_finite()
            && self.resistance.is_finite()
            && self.dt.is_finite()
            && self.v >= MAT_V_MIN
            && self.v <= MAT_V_MAX
            && self.v_reset >= MAT_V_MIN
            && self.v_reset <= MAT_V_MAX
            && self.theta1 >= 0.0_f64
            && self.theta1 <= MAT_THETA_MAX
            && self.theta2 >= 0.0_f64
            && self.theta2 <= MAT_THETA_MAX
            && self.h1 >= 0.0_f64
            && self.h1 <= MAT_THETA_MAX
            && self.h2 >= 0.0_f64
            && self.h2 <= MAT_THETA_MAX
            && self.tau_m > 0.0_f64
            && self.tau_1 > 0.0_f64
            && self.tau_2 > 0.0_f64
            && self.resistance > 0.0_f64
            && self.dt > 0.0_f64
    }

    fn derivatives(&self, v: f64, theta1: f64, theta2: f64, i_ext: f64) -> (f64, f64, f64) {
        let dv = (-(v - self.v_rest) + self.resistance * i_ext) / self.tau_m;
        (dv, -theta1 / self.tau_1, -theta2 / self.tau_2)
    }

    fn rk4_candidate(&self, v: f64, theta1: f64, theta2: f64, i_ext: f64) -> (f64, f64, f64) {
        let (k1v, k1t1, k1t2) = self.derivatives(v, theta1, theta2, i_ext);
        let (k2v, k2t1, k2t2) = self.derivatives(
            v + 0.5_f64 * self.dt * k1v,
            theta1 + 0.5_f64 * self.dt * k1t1,
            theta2 + 0.5_f64 * self.dt * k1t2,
            i_ext,
        );
        let (k3v, k3t1, k3t2) = self.derivatives(
            v + 0.5_f64 * self.dt * k2v,
            theta1 + 0.5_f64 * self.dt * k2t1,
            theta2 + 0.5_f64 * self.dt * k2t2,
            i_ext,
        );
        let (k4v, k4t1, k4t2) = self.derivatives(
            v + self.dt * k3v,
            theta1 + self.dt * k3t1,
            theta2 + self.dt * k3t2,
            i_ext,
        );
        let scale = self.dt / 6.0_f64;
        (
            v + scale * (k1v + 2.0_f64 * k2v + 2.0_f64 * k3v + k4v),
            theta1 + scale * (k1t1 + 2.0_f64 * k2t1 + 2.0_f64 * k3t1 + k4t1),
            theta2 + scale * (k1t2 + 2.0_f64 * k2t2 + 2.0_f64 * k3t2 + k4t2),
        )
    }
}

pub fn validate_mat(state: &MATNeuron) -> bool {
    state.validate()
}

fn mat_candidate_valid(v: f64, theta1: f64, theta2: f64) -> bool {
    v.is_finite()
        && theta1.is_finite()
        && theta2.is_finite()
        && v >= MAT_V_MIN
        && v <= MAT_V_MAX
        && theta1 >= 0.0_f64
        && theta1 <= MAT_THETA_MAX
        && theta2 >= 0.0_f64
        && theta2 <= MAT_THETA_MAX
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat_new() {
        let state = MATNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_mat(&state));
    }

    #[test]
    fn test_mat_step() {
        let mut state = MATNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_mat_rk4_candidate_commit() {
        let mut state = MATNeuron::new();
        state.theta1 = 0.5_f64;
        state.theta2 = 1.25_f64;
        let (v_candidate, theta1_candidate, theta2_candidate) =
            state.rk4_candidate(state.v, state.theta1, state.theta2, 10.0_f64);
        let spike = state.step(10.0_f64);
        assert_eq!(spike, 0);
        assert!((state.v - v_candidate).abs() < 1.0e-12_f64);
        assert!((state.theta1 - theta1_candidate).abs() < 1.0e-12_f64);
        assert!((state.theta2 - theta2_candidate).abs() < 1.0e-12_f64);
    }

    #[test]
    fn test_mat_invalid_state_does_not_mutate() {
        let mut state = MATNeuron::new();
        state.theta1 = -1.0_f64;
        let before = state.clone();
        assert_eq!(state.step(10.0_f64), -1);
        assert_eq!(state.v, before.v);
        assert_eq!(state.theta1, before.theta1);
        assert_eq!(state.theta2, before.theta2);
    }

    #[test]
    fn test_mat_spike_adds_threshold_candidates() {
        let mut state = MATNeuron::new();
        let (_, theta1_candidate, theta2_candidate) =
            state.rk4_candidate(state.v, state.theta1, state.theta2, 250.0_f64);
        assert_eq!(state.step(250.0_f64), 1);
        assert_eq!(state.v, state.v_reset);
        assert!((state.theta1 - (theta1_candidate + state.h1)).abs() < 1.0e-12_f64);
        assert!((state.theta2 - (theta2_candidate + state.h2)).abs() < 1.0e-12_f64);
    }
}
