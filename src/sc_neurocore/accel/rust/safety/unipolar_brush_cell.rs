// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for unipolar_brush_cell

#![allow(unused_variables, dead_code, non_snake_case)]

const V_MIN: f64 = -100.0;
const V_MAX: f64 = 60.0;

#[derive(Debug, Clone)]
pub struct UnipolarBrushCell {
    pub v: f64,
    pub persistent: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_persistent: f64,
    pub persistent_gain: f64,
    pub gain: f64,
    pub dt: f64,
}

impl UnipolarBrushCell {
    pub fn new() -> Self {
        Self {
            v: -65.0_f64,
            persistent: 0.0_f64,
            v_rest: -65.0_f64,
            v_reset: -70.0_f64,
            v_threshold: -50.0_f64,
            tau_m: 8.0_f64,
            tau_persistent: 200.0_f64,
            persistent_gain: 0.5_f64,
            gain: 2.5_f64,
            dt: 0.5_f64,
        }
    }

    fn finite(values: &[f64]) -> bool {
        values.iter().all(|value| value.is_finite())
    }

    fn valid_configuration(&self) -> bool {
        Self::finite(&[
            self.v_rest,
            self.v_reset,
            self.v_threshold,
            self.tau_m,
            self.tau_persistent,
            self.persistent_gain,
            self.gain,
            self.dt,
        ]) && self.tau_m > 0.0
            && self.tau_persistent > 0.0
            && self.persistent_gain >= 0.0
            && self.gain >= 0.0
            && self.dt > 0.0
            && self.v_reset < self.v_threshold
    }

    fn valid_state(&self) -> bool {
        Self::finite(&[self.v, self.persistent])
            && (V_MIN..=V_MAX).contains(&self.v)
            && self.persistent >= 0.0
    }

    fn first_order_relaxation(previous: f64, steady_state: f64, dt: f64, tau: f64) -> f64 {
        previous + (steady_state - previous) * (-(-dt / tau).exp_m1())
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !self.valid_configuration() || !self.valid_state() || !i_ext.is_finite() {
            return 0;
        }
        let input_drive = self.gain * i_ext.max(0.0);
        if !input_drive.is_finite() {
            return 0;
        }
        let next_persistent = Self::first_order_relaxation(
            self.persistent,
            self.persistent_gain * input_drive,
            self.dt,
            self.tau_persistent,
        )
        .max(0.0);
        let next_v = Self::first_order_relaxation(
            self.v,
            self.v_rest + input_drive + next_persistent,
            self.dt,
            self.tau_m,
        );
        if !Self::finite(&[next_persistent, next_v]) {
            return 0;
        }
        self.persistent = next_persistent;
        if next_v >= self.v_threshold {
            self.v = self.v_reset;
            return 1;
        }
        self.v = next_v.clamp(V_MIN, V_MAX);
        0
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.persistent = 0.0_f64;
    }
}

pub fn validate_unipolar_brush_cell(state: &UnipolarBrushCell) -> bool {
    state.valid_configuration() && state.valid_state()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_relaxation(previous: f64, steady_state: f64, dt: f64, tau: f64) -> f64 {
        previous + (steady_state - previous) * (-(-dt / tau).exp_m1())
    }

    #[test]
    fn test_unipolar_brush_cell_new() {
        let state = UnipolarBrushCell::new();
        assert!(state.v.is_finite());
        assert!(validate_unipolar_brush_cell(&state));
    }

    #[test]
    fn test_unipolar_brush_cell_step() {
        let mut state = UnipolarBrushCell::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn step_uses_closed_form_persistent_and_membrane_relaxation() {
        let mut state = UnipolarBrushCell::new();

        let spike = state.step(1.0);

        let input_drive = state.gain;
        let expected_persistent = exact_relaxation(
            0.0,
            state.persistent_gain * input_drive,
            state.dt,
            state.tau_persistent,
        );
        let expected_v = exact_relaxation(
            state.v_rest,
            state.v_rest + input_drive + expected_persistent,
            state.dt,
            state.tau_m,
        );
        assert_eq!(spike, 0);
        assert!((state.persistent - expected_persistent).abs() <= 1e-12);
        assert!((state.v - expected_v).abs() <= 1e-12);
    }

    #[test]
    fn invalid_current_preserves_state() {
        let mut state = UnipolarBrushCell::new();
        state.v = -63.0;
        state.persistent = 2.0;

        assert_eq!(state.step(f64::NAN), 0);

        assert_eq!(state.v, -63.0);
        assert_eq!(state.persistent, 2.0);
    }

    #[test]
    fn corrupted_state_preserved_on_step() {
        let mut state = UnipolarBrushCell::new();
        state.v = f64::NAN;
        state.persistent = 2.0;

        assert_eq!(state.step(10.0), 0);

        assert!(state.v.is_nan());
        assert_eq!(state.persistent, 2.0);
    }
}
