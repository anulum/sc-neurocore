// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
// Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

/// Safety surface for the nonlinear leaky integrate-and-fire neuron.

#[derive(Clone, Copy, Debug)]
pub struct NonlinearLifState {
    pub v: f64,
    pub w: f64,
    pub v_rest: f64,
    pub v_crit: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub a: f64,
    pub b: f64,
    pub tau_w: f64,
    pub c_m: f64,
    pub dt: f64,
}

impl Default for NonlinearLifState {
    fn default() -> Self {
        Self {
            v: -65.0,
            w: 0.0,
            v_rest: -65.0,
            v_crit: -40.0,
            v_threshold: -20.0,
            v_reset: -65.0,
            a: 0.04,
            b: 0.5,
            tau_w: 100.0,
            c_m: 1.0,
            dt: 0.1,
        }
    }
}

pub fn validate_nlif(state: &NonlinearLifState) -> bool {
    state.v.is_finite()
        && state.w.is_finite()
        && state.v_rest.is_finite()
        && state.v_crit.is_finite()
        && state.v_threshold.is_finite()
        && state.v_reset.is_finite()
        && state.a.is_finite()
        && state.b.is_finite()
        && state.tau_w.is_finite()
        && state.c_m.is_finite()
        && state.dt.is_finite()
        && state.v_rest < state.v_crit
        && state.v_crit < state.v_threshold
        && state.v_reset < state.v_threshold
        && state.a >= 0.0
        && state.b >= 0.0
        && state.tau_w > 0.0
        && state.c_m > 0.0
        && state.dt > 0.0
        && state.dt <= state.tau_w
}

fn derivatives(state: &NonlinearLifState, v: f64, w: f64, current: f64) -> (f64, f64) {
    let nonlinear = state.a * (v - state.v_rest) * (v - state.v_crit);
    let dv = (nonlinear - w + current) / state.c_m;
    let dw = (state.b * (v - state.v_rest) - w) / state.tau_w;
    (dv, dw)
}

fn rk4_candidate(state: &NonlinearLifState, current: f64) -> Option<(f64, f64)> {
    let (k1v, k1w) = derivatives(state, state.v, state.w, current);
    let (k2v, k2w) = derivatives(
        state,
        state.v + 0.5 * state.dt * k1v,
        state.w + 0.5 * state.dt * k1w,
        current,
    );
    let (k3v, k3w) = derivatives(
        state,
        state.v + 0.5 * state.dt * k2v,
        state.w + 0.5 * state.dt * k2w,
        current,
    );
    let (k4v, k4w) = derivatives(
        state,
        state.v + state.dt * k3v,
        state.w + state.dt * k3w,
        current,
    );
    let next_v = state.v + (state.dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    let next_w = state.w + (state.dt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w);
    if next_v.is_finite() && next_w.is_finite() {
        Some((next_v, next_w))
    } else {
        None
    }
}

pub fn step(state: &mut NonlinearLifState, current: f64) -> i32 {
    if !current.is_finite() || !validate_nlif(state) {
        return -1;
    }

    let Some((next_v, next_w)) = rk4_candidate(state, current) else {
        return -1;
    };
    state.v = next_v;
    state.w = next_w;

    if next_v >= state.v_threshold {
        state.v = state.v_reset;
        1
    } else {
        0
    }
}

pub fn reset(state: &mut NonlinearLifState) {
    state.v = state.v_rest;
    state.w = 0.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_voltage_geometry() {
        let mut state = NonlinearLifState::default();
        assert!(validate_nlif(&state));
        state.v_crit = -70.0;
        assert!(!validate_nlif(&state));
    }

    #[test]
    fn rejects_non_finite_current_before_mutation() {
        let mut state = NonlinearLifState {
            v: -60.0,
            w: 0.5,
            ..NonlinearLifState::default()
        };
        assert_eq!(step(&mut state, f64::NAN), -1);
        assert_eq!(state.v, -60.0);
        assert_eq!(state.w, 0.5);
    }

    #[test]
    fn reset_preserves_parameters() {
        let mut state = NonlinearLifState {
            v: -40.0,
            w: 2.0,
            v_rest: -62.0,
            v_reset: -58.0,
            ..NonlinearLifState::default()
        };
        reset(&mut state);
        assert_eq!(state.v, -62.0);
        assert_eq!(state.w, 0.0);
        assert_eq!(state.v_reset, -58.0);
    }
}
