// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for astrocyte

//! Li-Rinzel astrocyte calcium model: the standalone safety mirror of the Python reference
//! `sc_neurocore.neurons.models.astrocyte` and the engine's `AstrocyteModel`.

/// Li-Rinzel IP3R astrocyte: cytosolic Ca²⁺, the IP3-receptor gate and IP3.
#[derive(Debug, Clone)]
pub struct AstrocyteModel {
    /// Cytosolic calcium (uM).
    pub ca: f64,
    /// IP3-receptor de-inactivation gate (dimensionless, in 0 to 1).
    pub h: f64,
    /// IP3 concentration (uM).
    pub ip3: f64,
    /// Maximal calcium flux through the IP3 receptor (1/s).
    pub v_er: f64,
    /// SERCA pump half-activation calcium concentration (uM).
    pub k_er: f64,
    /// Maximal SERCA pump uptake rate (uM/s).
    pub v_serca: f64,
    /// IP3 dissociation constant of the IP3 receptor (uM).
    pub d1: f64,
    /// Calcium-inhibition dissociation constant (uM).
    pub d2: f64,
    /// IP3 dissociation constant (inhibitory site) (uM).
    pub d3: f64,
    /// Calcium-activation dissociation constant (uM).
    pub d5: f64,
    /// Calcium-inactivation rate of the receptor gate (1/(uM s)).
    pub a2: f64,
    /// Total cell calcium (cytosol plus ER) (uM).
    pub c0: f64,
    /// ER-to-cytosol volume ratio (dimensionless).
    pub c1: f64,
    /// ER leak rate (1/s).
    pub leak: f64,
    /// Constant external IP3 production (uM/s), added to the input.
    pub ip3_prod: f64,
    /// IP3 decay rate (1/s).
    pub ip3_decay: f64,
    /// Euler step (s).
    pub dt: f64,
}

impl AstrocyteModel {
    /// Create a cell with the Python reference's defaults: `ca` 0.05, `h` 0.8, `ip3` 0.5.
    pub fn new() -> Self {
        Self {
            ca: 0.05,
            h: 0.8,
            ip3: 0.5,
            v_er: 0.9,
            k_er: 0.15,
            v_serca: 0.4,
            d1: 0.13,
            d2: 1.049,
            d3: 0.9434,
            d5: 0.08234,
            a2: 0.2,
            c0: 2.0,
            c1: 0.185,
            leak: 0.01,
            ip3_prod: 0.0,
            ip3_decay: 0.14,
            dt: 0.01,
        }
    }

    /// Advance one forward-Euler step with `i_ext`, the input IP3 production (uM/s), and
    /// return the new cytosolic calcium. The recurrence is the Python reference's, term for
    /// term, with every term from the old state.
    ///
    /// # Errors
    ///
    /// A negative or non-finite input, an invalid parameter or state (calcium may equal
    /// `c0` after a step), or a calcium candidate that is not finite or exceeds `c0` returns
    /// an error and leaves the state unchanged.
    pub fn step(&mut self, i_ext: f64) -> Result<f64, &'static str> {
        if !i_ext.is_finite() || i_ext < 0.0 {
            return Err("current must be finite and non-negative");
        }
        let at_pool_edge = self.ca == self.c0;
        if !(validate_astrocyte(self) || (at_pool_edge && validate_parameters(self))) {
            return Err("astrocyte parameters and state must be finite and in range");
        }
        let m_inf = self.ip3 / (self.ip3 + self.d1);
        let n_inf = self.ca / (self.ca + self.d5);
        let ca_er = (self.c0 - self.ca) / self.c1;
        let j_channel = self.v_er * (m_inf * n_inf * self.h).powf(3.0) * (ca_er - self.ca);
        let j_serca = self.v_serca * self.ca.powf(2.0) / (self.ca.powf(2.0) + self.k_er.powf(2.0));
        let j_leak = self.leak * (ca_er - self.ca);
        let dca = j_channel - j_serca + j_leak;
        let q2 = self.d2 * (self.ip3 + self.d1) / (self.ip3 + self.d3);
        let h_inf = q2 / (q2 + self.ca);
        let tau_h = 1.0 / (self.a2 * (q2 + self.ca));
        let dh = (h_inf - self.h) / tau_h.max(1e-6);
        let dip3 = i_ext + self.ip3_prod - self.ip3_decay * self.ip3;
        let ca_next = self.ca + dca * self.dt;
        if !ca_next.is_finite() || ca_next > self.c0 {
            return Err("calcium update must remain finite and within the total calcium pool");
        }
        self.ca = ca_next.max(0.0);
        self.h = (self.h + dh * self.dt).clamp(0.0, 1.0);
        self.ip3 = (self.ip3 + dip3 * self.dt).max(0.0);
        Ok(self.ca)
    }

    /// Restore `ca`, `h` and `ip3` to 0.05, 0.8 and 0.5; the parameters are kept.
    pub fn reset(&mut self) {
        self.ca = 0.05;
        self.h = 0.8;
        self.ip3 = 0.5;
    }
}

impl Default for AstrocyteModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Return whether every parameter, and the state apart from the pool bound, is valid.
fn validate_parameters(state: &AstrocyteModel) -> bool {
    let mut probe = state.clone();
    probe.ca = 0.0;
    validate_astrocyte(&probe) && state.ca.is_finite()
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
        let ca = state.step(10.0).unwrap();
        assert!((0.0..=state.c0).contains(&ca));
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

    #[test]
    fn input_drives_ip3_as_the_python_reference() {
        // Python reference after 10 steps at 2.5: ca 0.075363, ip3 0.741475.
        let mut state = AstrocyteModel::new();
        for _ in 0..10 {
            state.step(2.5).unwrap();
        }
        assert!((state.ip3 - 0.741475).abs() < 5e-7, "ip3 {}", state.ip3);
        assert!((state.ca - 0.075363).abs() < 5e-7, "ca {}", state.ca);
    }

    #[test]
    fn refuses_invalid_input_and_keeps_the_state() {
        for value in [-0.1, f64::NAN, f64::INFINITY] {
            let mut state = AstrocyteModel::new();
            let before = (state.ca, state.h, state.ip3);
            assert!(state.step(value).is_err());
            assert_eq!((state.ca, state.h, state.ip3), before);
        }
    }

    #[test]
    fn refuses_a_calcium_candidate_beyond_the_pool() {
        let mut state = AstrocyteModel::new();
        state.h = 1.0;
        state.ip3 = 50.0;
        state.v_er = 1.0e9;
        let before = (state.ca, state.h, state.ip3);
        assert!(state.step(0.0).is_err());
        assert_eq!((state.ca, state.h, state.ip3), before);
    }

    #[test]
    fn steps_from_calcium_at_the_pool_edge() {
        // The Python reference checks ca < c0 only at construction, so a step may start at c0.
        let mut state = AstrocyteModel::new();
        state.ca = state.c0;
        assert!(state.step(0.0).is_ok());
    }

    #[test]
    fn reset_keeps_the_parameters() {
        let mut state = AstrocyteModel::new();
        state.v_er = 1.5;
        state.step(1.0).unwrap();
        state.reset();
        assert_eq!(
            (state.ca, state.h, state.ip3, state.v_er),
            (0.05, 0.8, 0.5, 1.5)
        );
    }
}
