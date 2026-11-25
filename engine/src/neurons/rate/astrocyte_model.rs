// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Li-Rinzel astrocyte model

#![deny(missing_docs)]

/// Li-Rinzel IP3R astrocyte model: cytosolic Ca²⁺, the IP3-receptor gate and IP3.
///
/// The same recurrence as the Python reference `sc_neurocore.neurons.models.astrocyte`:
/// three forward-Euler ODEs, an ER leak at rate `leak`, and an input that drives IP3
/// production (glutamate-driven), not calcium.
#[derive(Clone, Debug)]
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
    /// Euler step (s); the rates are per second.
    pub dt: f64,
}

impl AstrocyteModel {
    /// Create a cell with `ca` = 0.05 uM, `h` = 0.8 and `ip3` = 0.5 uM, the Python
    /// reference's default parameters, and `dt` = 0.01 s.
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

    /// Return whether every parameter and the state are finite and in range; calcium may equal
    /// `c0`, which a step can reach.
    fn valid_runtime(&self) -> bool {
        let positive = [
            self.v_er,
            self.k_er,
            self.v_serca,
            self.d1,
            self.d2,
            self.d3,
            self.d5,
            self.a2,
            self.c0,
            self.c1,
            self.dt,
        ];
        let non_negative = [self.leak, self.ip3_prod, self.ip3_decay];
        positive
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
            && non_negative
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
            && self.ca.is_finite()
            && self.ca >= 0.0
            && self.ca <= self.c0
            && self.h.is_finite()
            && (0.0..=1.0).contains(&self.h)
            && self.ip3.is_finite()
            && self.ip3 >= 0.0
    }

    /// Advance one forward-Euler step with `current`, the input IP3 production (uM/s), and
    /// return the new cytosolic calcium.
    ///
    /// With `ca_er = (c0 - ca) / c1`, the calcium change is the channel flux
    /// `v_er * (m_inf * n_inf * h)^3 * (ca_er - ca)` minus the pump
    /// `v_serca * ca^2 / (ca^2 + k_er^2)` plus the leak `leak * (ca_er - ca)`. The gate relaxes
    /// towards `q2 / (q2 + ca)` with time constant `1 / (a2 * (q2 + ca))` (at least 1e-6 s),
    /// and `dip3 = current + ip3_prod - ip3_decay * ip3`. Every term uses the old state. Then
    /// `ca` is floored at 0, `h` is clipped to 0 to 1 and `ip3` is floored at 0.
    ///
    /// # Errors
    ///
    /// A negative or non-finite `current`, an invalid parameter or state, or a calcium
    /// candidate that is not finite or exceeds `c0` returns an error and leaves the state
    /// unchanged.
    pub fn try_step(&mut self, current: f64) -> Result<f64, &'static str> {
        if !current.is_finite() || current < 0.0 {
            return Err("current must be finite and non-negative");
        }
        if !self.valid_runtime() {
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
        let dip3 = current + self.ip3_prod - self.ip3_decay * self.ip3;
        let ca_next = self.ca + dca * self.dt;
        if !ca_next.is_finite() || ca_next > self.c0 {
            return Err("calcium update must remain finite and within the total calcium pool");
        }
        self.ca = ca_next.max(0.0);
        self.h = (self.h + dh * self.dt).clamp(0.0, 1.0);
        self.ip3 = (self.ip3 + dip3 * self.dt).max(0.0);
        Ok(self.ca)
    }

    /// Advance one step as [`AstrocyteModel::try_step`] does and return the calcium; a refused
    /// step leaves the state unchanged and returns the current calcium.
    pub fn step(&mut self, current: f64) -> f64 {
        self.try_step(current).unwrap_or(self.ca)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn astrocyte_ca() {
        let mut n = AstrocyteModel::new();
        let mut max_ca = 0.0_f64;
        for _ in 0..5000 {
            let c = n.step(0.1);
            max_ca = max_ca.max(c);
        }
        assert!(max_ca > 0.05);
    }

    #[test]
    fn astrocyte_reset() {
        let mut n = AstrocyteModel::new();
        for _ in 0..1000 {
            n.step(0.1);
        }
        n.reset();
        assert!((n.ca - 0.05).abs() < 1e-10);
    }

    #[test]
    fn astrocyte_nan_no_panic() {
        AstrocyteModel::new().step(f64::NAN);
    }

    #[test]
    fn astrocyte_input_drives_ip3_as_the_python_reference() {
        // Python reference after 10 steps at 2.5: ca 0.075363, ip3 0.741475.
        let mut n = AstrocyteModel::new();
        for _ in 0..10 {
            n.try_step(2.5).unwrap();
        }
        assert!((n.ip3 - 0.741475).abs() < 5e-7, "ip3 {}", n.ip3);
        assert!((n.ca - 0.075363).abs() < 5e-7, "ca {}", n.ca);
    }

    #[test]
    fn astrocyte_refuses_invalid_input_and_keeps_state() {
        for current in [-0.1, f64::NAN, f64::INFINITY] {
            let mut n = AstrocyteModel::new();
            let before = (n.ca, n.h, n.ip3);
            assert!(n.try_step(current).is_err());
            assert_eq!(n.step(current), before.0);
            assert_eq!((n.ca, n.h, n.ip3), before);
        }
    }

    #[test]
    fn astrocyte_refuses_a_calcium_candidate_beyond_the_pool() {
        // A huge channel flux from the default calcium overshoots the 2 uM pool in one step.
        let mut n = AstrocyteModel::new();
        n.h = 1.0;
        n.ip3 = 50.0;
        n.v_er = 1.0e9;
        let before = (n.ca, n.h, n.ip3);
        assert!(n.try_step(0.0).is_err());
        assert_eq!((n.ca, n.h, n.ip3), before);
    }

    #[test]
    fn astrocyte_refuses_invalid_parameters() {
        let mut n = AstrocyteModel::new();
        n.leak = -1.0;
        assert!(n.try_step(0.0).is_err());
        let mut n = AstrocyteModel::new();
        n.dt = 0.0;
        assert!(n.try_step(0.0).is_err());
    }

    #[test]
    fn astrocyte_ca_nonneg() {
        let mut n = AstrocyteModel::new();
        for _ in 0..5000 {
            n.step(0.1);
        }
        assert!(n.ca >= 0.0, "Ca²⁺ must be non-negative");
    }
}
