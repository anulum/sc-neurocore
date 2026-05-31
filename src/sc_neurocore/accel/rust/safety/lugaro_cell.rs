// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for lugaro_cell

#[derive(Debug, Clone)]
pub struct LugaroCell {
    pub v: f64,
    pub adapt: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_adapt: f64,
    pub a_adapt: f64,
    pub gain: f64,
    pub serotonin: f64,
    pub dt: f64,
}

impl Default for LugaroCell {
    fn default() -> Self {
        Self::new()
    }
}

impl LugaroCell {
    pub fn new() -> Self {
        Self {
            v: -55.0,
            adapt: 0.0,
            v_rest: -55.0,
            v_reset: -65.0,
            v_threshold: -48.0,
            tau_m: 10.0,
            tau_adapt: 150.0,
            a_adapt: 0.05,
            gain: 2.0,
            serotonin: 0.0,
            dt: 0.5,
        }
    }

    pub fn with_serotonin(level: f64) -> Self {
        let mut state = Self::new();
        state.serotonin = level.clamp(0.0, 1.0);
        state
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_lugaro_cell(self) || !i_ext.is_finite() {
            return 0;
        }

        let effective_gain = self.gain * (1.0 + 0.5 * self.serotonin);
        let input = effective_gain * i_ext;
        let v_inf = self.v_rest + input - self.adapt;
        let v_next = v_inf + (self.v - v_inf) * (-self.dt / self.tau_m).exp();
        let adapt_inf = (self.a_adapt * (v_next - self.v_rest).max(0.0)).max(0.0);
        let adapt_next =
            (adapt_inf + (self.adapt - adapt_inf) * (-self.dt / self.tau_adapt).exp()).max(0.0);
        if !v_next.is_finite() || !adapt_next.is_finite() {
            return 0;
        }

        if v_next >= self.v_threshold {
            self.v = self.v_reset;
            self.adapt = adapt_next + 1.0;
            return 1;
        }

        self.v = v_next.clamp(-100.0, 60.0);
        self.adapt = adapt_next;
        0
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.adapt = 0.0;
    }
}

pub fn validate_lugaro_cell(state: &LugaroCell) -> bool {
    [
        state.v,
        state.adapt,
        state.v_rest,
        state.v_reset,
        state.v_threshold,
        state.tau_m,
        state.tau_adapt,
        state.a_adapt,
        state.gain,
        state.serotonin,
        state.dt,
    ]
    .iter()
    .all(|value| value.is_finite())
        && state.tau_m > 0.0
        && state.tau_adapt > 0.0
        && state.dt > 0.0
        && state.a_adapt >= 0.0
        && state.gain >= 0.0
        && (-100.0..=60.0).contains(&state.v)
        && (0.0..=1.0).contains(&state.serotonin)
        && state.adapt >= 0.0
        && state.v_threshold > state.v_reset
        && state.v_threshold > state.v_rest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lugaro_cell_new() {
        let state = LugaroCell::new();
        assert!(state.v.is_finite());
        assert!(validate_lugaro_cell(&state));
    }

    #[test]
    fn test_lugaro_cell_step() {
        let mut state = LugaroCell::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_lugaro_cell_serotonin_raises_firing() {
        let mut without = LugaroCell::new();
        let mut with = LugaroCell::with_serotonin(1.0);
        let mut spikes_without = 0;
        let mut spikes_with = 0;
        for _ in 0..2000 {
            spikes_without += without.step(3.0);
            spikes_with += with.step(3.0);
        }
        assert!(spikes_with >= spikes_without);
    }

    #[test]
    fn test_lugaro_cell_invalid_drive_preserves_state() {
        let mut state = LugaroCell::new();
        let before = state.clone();
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!(state.v, before.v);
        assert_eq!(state.adapt, before.adapt);
    }

    #[test]
    fn test_lugaro_cell_corrupted_state_preserves_state() {
        let mut state = LugaroCell::new();
        state.adapt = f64::NAN;
        let before = state.clone();
        assert_eq!(state.step(5.0), 0);
        assert_eq!(state.v, before.v);
        assert!(state.adapt.is_nan());
    }

    #[test]
    fn test_lugaro_cell_invalid_voltage_preserves_state() {
        let mut state = LugaroCell::new();
        state.v = 60.1;
        let before = state.clone();
        assert_eq!(state.step(5.0), 0);
        assert_eq!(state.v, before.v);
        assert_eq!(state.adapt, before.adapt);
    }

    #[test]
    fn test_lugaro_cell_closed_form_membrane_and_adaptation_relaxation() {
        let mut state = LugaroCell::new();
        state.v = -56.0;
        state.adapt = 0.2;
        state.gain = 0.0;

        let v_inf = state.v_rest - state.adapt;
        let expected_v = exact_relax_lugaro(state.v, v_inf, state.tau_m, state.dt);
        let adapt_inf = (state.a_adapt * (expected_v - state.v_rest).max(0.0)).max(0.0);
        let expected_adapt =
            exact_relax_lugaro(state.adapt, adapt_inf, state.tau_adapt, state.dt).max(0.0);

        assert_eq!(state.step(0.0), 0);
        assert_close_lugaro(state.v, expected_v, 1e-12);
        assert_close_lugaro(state.adapt, expected_adapt, 1e-12);
    }

    fn exact_relax_lugaro(value: f64, target: f64, tau: f64, dt: f64) -> f64 {
        target + (value - target) * (-dt / tau).exp()
    }

    fn assert_close_lugaro(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={:.16e} expected={:.16e} tolerance={:.3e}",
            actual,
            expected,
            tolerance
        );
    }
}
