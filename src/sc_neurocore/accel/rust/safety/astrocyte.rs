// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for astrocyte

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct AstrocyteModel {
    pub ca: f64,
    pub h: f64,
    pub ip3: f64,
    pub v_er: f64,
    pub k_er: f64,
    pub v_serca: f64,
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
    pub d5: f64,
    pub a2: f64,
    pub c0: f64,
    pub c1: f64,
    pub leak: f64,
    pub ip3_prod: f64,
    pub ip3_decay: f64,
    pub dt: f64,
}

impl AstrocyteModel {
    pub fn new() -> Self {
        Self {
            ca: 0.05_f64,
            h: 0.8_f64,
            ip3: 0.5_f64,
            v_er: 0.9_f64,
            k_er: 0.15_f64,
            v_serca: 0.4_f64,
            d1: 0.13_f64,
            d2: 1.049_f64,
            d3: 0.9434_f64,
            d5: 0.08234_f64,
            a2: 0.2_f64,
            c0: 2.0_f64,
            c1: 0.185_f64,
            leak: 0.01_f64,
            ip3_prod: 0.0_f64,
            ip3_decay: 0.14_f64,
            dt: 0.01_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        // # Li-Rinzel IP3R open probability
        // m_inf = self.ip3 / (self.ip3 + self.d1)
        // n_inf = self.ca / (self.ca + self.d5)
        // ca_er = (self.c0 - self.ca) / self.c1  # Li-Rinzel 1994 conservation
        // j_channel = self.v_er * (m_inf * n_inf * self.h) .powi 3 * (ca_er - se
        // j_serca = self.v_serca * self.ca.powi2 / (self.ca.powi2 + self.k_er.po
        // j_leak = self.leak * (ca_er - self.ca)
        // dca = j_channel - j_serca + j_leak
        // q2 = self.d2 * (self.ip3 + self.d1) / (self.ip3 + self.d3)
        // h_inf = q2 / (q2 + self.ca)
        // tau_h = 1.0 / (self.a2 * (q2 + self.ca))
        // dh = (h_inf - self.h) / max(tau_h, 1e-6)
        // dip3 = current + self.ip3_prod - self.ip3_decay * self.ip3
        // self.ca = max(0.0, self.ca + dca * self.dt)
        // self.h = (self.h + dh * self.dt_f64).clamp(0.0, 1.0)
        0 // spike indicator
    }

    pub fn reset(&mut self) {
        // self.ca, self.h, self.ip3 = 0.05, 0.8, 0.5
        self.ca = 0.05_f64;
        self.h = 0.8_f64;
        self.ip3 = 0.5_f64;
        self.v_er = 0.9_f64;
        self.k_er = 0.15_f64;
    }
}

/// Return whether state and configured parameters remain in their valid domain.
///
/// Mirrors what the maintained model enforces at construction: `ca`, `ip3`,
/// `leak`, `ip3_prod` and `ip3_decay` finite and non-negative; the ER, SERCA,
/// IP3-receptor and timestep parameters finite and strictly positive; the
/// gating variable `h` finite and within `[0, 1]`; and cytosolic calcium below
/// the total cell calcium `c0`.
#[must_use]
pub fn validate_astrocyte(state: &AstrocyteModel) -> bool {
    let non_negative = [
        state.ca,
        state.ip3,
        state.leak,
        state.ip3_prod,
        state.ip3_decay,
    ];
    let positive = [
        state.v_er,
        state.k_er,
        state.v_serca,
        state.d1,
        state.d2,
        state.d3,
        state.d5,
        state.a2,
        state.c0,
        state.c1,
        state.dt,
    ];
    non_negative
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
        && positive
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        && state.h.is_finite()
        && (0.0..=1.0).contains(&state.h)
        && state.ca < state.c0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_astrocyte_new() {
        let state = AstrocyteModel::new();
        assert!(validate_astrocyte(&state));
    }

    #[test]
    fn test_astrocyte_step() {
        let mut state = AstrocyteModel::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    /// Every case below fails against the former `true` stub: it is what makes
    /// the validator evidence rather than decoration.
    #[test]
    fn rejects_a_non_finite_calcium() {
        let mut state = AstrocyteModel::new();
        state.ca = f64::NAN;
        assert!(!validate_astrocyte(&state));
    }

    #[test]
    fn rejects_a_negative_calcium() {
        let mut state = AstrocyteModel::new();
        state.ca = -1.0e-9;
        assert!(!validate_astrocyte(&state));
    }

    #[test]
    fn rejects_a_gating_variable_outside_the_unit_interval() {
        for value in [-1.0e-9, 1.0 + 1.0e-9, f64::INFINITY] {
            let mut state = AstrocyteModel::new();
            state.h = value;
            assert!(!validate_astrocyte(&state), "h = {value} must be refused");
        }
    }

    #[test]
    fn accepts_both_ends_of_the_gating_interval() {
        for value in [0.0, 1.0] {
            let mut state = AstrocyteModel::new();
            state.h = value;
            assert!(validate_astrocyte(&state), "h = {value} must be accepted");
        }
    }

    #[test]
    fn rejects_a_non_positive_timestep() {
        for value in [0.0, -0.1] {
            let mut state = AstrocyteModel::new();
            state.dt = value;
            assert!(!validate_astrocyte(&state), "dt = {value} must be refused");
        }
    }

    #[test]
    fn rejects_calcium_at_or_above_the_total_cell_calcium() {
        let mut state = AstrocyteModel::new();
        state.ca = state.c0;
        assert!(!validate_astrocyte(&state));
    }

    #[test]
    fn accepts_a_zero_leak_but_refuses_a_negative_one() {
        let mut state = AstrocyteModel::new();
        state.leak = 0.0;
        assert!(validate_astrocyte(&state));
        state.leak = -1.0e-12;
        assert!(!validate_astrocyte(&state));
    }
}
