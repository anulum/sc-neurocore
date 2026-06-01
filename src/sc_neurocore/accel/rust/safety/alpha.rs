// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for alpha

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct AlphaNeuron {
    pub v: f64,
    pub a_exc: f64,
    pub i_exc: f64,
    pub a_inh: f64,
    pub i_inh: f64,
    pub v_rest: f64,
    pub v_threshold: f64,
    pub tau_v: f64,
    pub tau_exc: f64,
    pub tau_inh: f64,
    pub dt: f64,
}

impl AlphaNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0_f64,
            a_exc: 0.0_f64,
            i_exc: 0.0_f64,
            a_inh: 0.0_f64,
            i_inh: 0.0_f64,
            v_rest: 0.0_f64,
            v_threshold: 1.0_f64,
            tau_v: 20.0_f64,
            tau_exc: 5.0_f64,
            tau_inh: 10.0_f64,
            dt: 1.0_f64,
        }
    }

    pub fn step(&mut self, exc_current: f64, inh_current: f64) -> Result<i32, &'static str> {
        if !exc_current.is_finite() || !inh_current.is_finite() {
            return Err("alpha currents must be finite");
        }
        validate_alpha(self)?;

        let exc_steady = self.tau_exc * exc_current;
        let inh_steady = self.tau_inh * inh_current;
        let exc_rise_delta = self.a_exc - exc_steady;
        let inh_rise_delta = self.a_inh - inh_steady;
        let exc_current_delta = self.i_exc - exc_steady;
        let inh_current_delta = self.i_inh - inh_steady;
        let (a_exc_next, i_exc_next) =
            alpha_filter_candidates(self.a_exc, self.i_exc, exc_current, self.tau_exc, self.dt);
        let (a_inh_next, i_inh_next) =
            alpha_filter_candidates(self.a_inh, self.i_inh, inh_current, self.tau_inh, self.dt);
        let v_steady = self.v_rest + exc_steady - inh_steady;
        let v_next = v_steady
            + (self.v - v_steady) * (-self.dt / self.tau_v).exp()
            + alpha_membrane_drive_contribution(
                exc_current_delta,
                exc_rise_delta,
                self.tau_exc,
                self.tau_v,
                self.dt,
            )
            - alpha_membrane_drive_contribution(
                inh_current_delta,
                inh_rise_delta,
                self.tau_inh,
                self.tau_v,
                self.dt,
            );
        if !a_exc_next.is_finite()
            || !i_exc_next.is_finite()
            || !a_inh_next.is_finite()
            || !i_inh_next.is_finite()
            || !v_next.is_finite()
        {
            return Err("alpha exact-flow update became non-finite");
        }

        self.a_exc = a_exc_next;
        self.i_exc = i_exc_next;
        self.a_inh = a_inh_next;
        self.i_inh = i_inh_next;
        if v_next >= self.v_threshold {
            self.v = self.v_rest;
            return Ok(1);
        }
        self.v = v_next;
        Ok(0)
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.a_exc = 0.0_f64;
        self.i_exc = 0.0_f64;
        self.a_inh = 0.0_f64;
        self.i_inh = 0.0_f64;
    }
}

fn alpha_filter_candidates(
    rise_state: f64,
    current_state: f64,
    drive: f64,
    tau: f64,
    dt: f64,
) -> (f64, f64) {
    let steady_state = tau * drive;
    let rise_delta = rise_state - steady_state;
    let current_delta = current_state - steady_state;
    let decay = (-dt / tau).exp();
    let rise_next = steady_state + rise_delta * decay;
    let current_next = steady_state + decay * (current_delta + rise_delta * dt / tau);
    (rise_next, current_next)
}

fn alpha_membrane_drive_contribution(
    current_delta: f64,
    rise_delta: f64,
    tau_drive: f64,
    tau_v: f64,
    dt: f64,
) -> f64 {
    let rate_v = 1.0 / tau_v;
    let rate_drive = 1.0 / tau_drive;
    let decay_v = (-dt / tau_v).exp();
    let decay_drive = (-dt / tau_drive).exp();
    if (rate_v - rate_drive).abs() <= 1.0e-14 {
        return rate_v * decay_v * (current_delta * dt + rise_delta * dt * dt / (2.0 * tau_drive));
    }
    let rate_delta = rate_v - rate_drive;
    let first_order = current_delta * (decay_drive - decay_v) / rate_delta;
    let second_order = rise_delta / tau_drive * (decay_drive * (rate_delta * dt - 1.0) + decay_v)
        / (rate_delta * rate_delta);
    rate_v * (first_order + second_order)
}

pub fn validate_alpha(state: &AlphaNeuron) -> Result<(), &'static str> {
    if state.v.is_finite()
        && state.a_exc.is_finite()
        && state.i_exc.is_finite()
        && state.a_inh.is_finite()
        && state.i_inh.is_finite()
        && state.v_rest.is_finite()
        && state.v_threshold.is_finite()
        && state.tau_v.is_finite()
        && state.tau_v > 0.0
        && state.tau_exc.is_finite()
        && state.tau_exc > 0.0
        && state.tau_inh.is_finite()
        && state.tau_inh > 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
    {
        Ok(())
    } else {
        Err("alpha state variables, time constants, and timestep must be finite and valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drive_contribution(
        current_delta: f64,
        rise_delta: f64,
        tau_drive: f64,
        tau_v: f64,
        dt: f64,
    ) -> f64 {
        let rate_v = 1.0 / tau_v;
        let rate_drive = 1.0 / tau_drive;
        let decay_v = (-dt / tau_v).exp();
        let decay_drive = (-dt / tau_drive).exp();
        if (rate_v - rate_drive).abs() <= 1.0e-14 {
            return rate_v
                * decay_v
                * (current_delta * dt + rise_delta * dt * dt / (2.0 * tau_drive));
        }
        let rate_delta = rate_v - rate_drive;
        let first_order = current_delta * (decay_drive - decay_v) / rate_delta;
        let second_order = rise_delta / tau_drive
            * (decay_drive * (rate_delta * dt - 1.0) + decay_v)
            / (rate_delta * rate_delta);
        rate_v * (first_order + second_order)
    }

    fn exact_reference(
        state: &AlphaNeuron,
        exc_current: f64,
        inh_current: f64,
    ) -> (f64, f64, f64, f64, f64) {
        let a_exc_ss = state.tau_exc * exc_current;
        let a_inh_ss = state.tau_inh * inh_current;
        let a_exc_delta = state.a_exc - a_exc_ss;
        let a_inh_delta = state.a_inh - a_inh_ss;
        let i_exc_delta = state.i_exc - a_exc_ss;
        let i_inh_delta = state.i_inh - a_inh_ss;
        let decay_exc = (-state.dt / state.tau_exc).exp();
        let decay_inh = (-state.dt / state.tau_inh).exp();
        let a_exc_next = a_exc_ss + a_exc_delta * decay_exc;
        let a_inh_next = a_inh_ss + a_inh_delta * decay_inh;
        let i_exc_next =
            a_exc_ss + decay_exc * (i_exc_delta + a_exc_delta * state.dt / state.tau_exc);
        let i_inh_next =
            a_inh_ss + decay_inh * (i_inh_delta + a_inh_delta * state.dt / state.tau_inh);
        let v_steady = state.v_rest + a_exc_ss - a_inh_ss;
        let v_next = v_steady
            + (state.v - v_steady) * (-state.dt / state.tau_v).exp()
            + drive_contribution(
                i_exc_delta,
                a_exc_delta,
                state.tau_exc,
                state.tau_v,
                state.dt,
            )
            - drive_contribution(
                i_inh_delta,
                a_inh_delta,
                state.tau_inh,
                state.tau_v,
                state.dt,
            );
        (v_next, a_exc_next, i_exc_next, a_inh_next, i_inh_next)
    }

    #[test]
    fn test_alpha_new() {
        let state = AlphaNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_alpha(&state).is_ok());
    }

    #[test]
    fn test_alpha_step() {
        let mut state = AlphaNeuron::new();
        let spike = state.step(10.0, 0.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn exact_linear_flow_matches_reference() {
        let mut state = AlphaNeuron::new();
        state.v = 0.3;
        state.a_exc = 0.9;
        state.i_exc = 0.7;
        state.a_inh = 0.25;
        state.i_inh = 0.2;
        state.v_threshold = 100.0;
        state.dt = 0.75;
        let (expected_v, expected_a_exc, expected_i_exc, expected_a_inh, expected_i_inh) =
            exact_reference(&state, 0.8, 0.1);

        let spike = state.step(0.8, 0.1).unwrap();

        assert_eq!(spike, 0);
        assert!((state.v - expected_v).abs() < 1.0e-12);
        assert!((state.a_exc - expected_a_exc).abs() < 1.0e-12);
        assert!((state.i_exc - expected_i_exc).abs() < 1.0e-12);
        assert!((state.a_inh - expected_a_inh).abs() < 1.0e-12);
        assert!((state.i_inh - expected_i_inh).abs() < 1.0e-12);
    }

    #[test]
    fn invalid_state_does_not_mutate() {
        let mut state = AlphaNeuron::new();
        state.v = 0.25;
        state.a_exc = 0.6;
        state.i_exc = 0.5;
        state.a_inh = 0.2;
        state.i_inh = 0.125;
        let before = (state.v, state.a_exc, state.i_exc, state.a_inh, state.i_inh);
        state.tau_v = 0.0;

        assert!(state.step(1.0, 0.5).is_err());
        assert_eq!(
            (state.v, state.a_exc, state.i_exc, state.a_inh, state.i_inh),
            before
        );
    }
}
