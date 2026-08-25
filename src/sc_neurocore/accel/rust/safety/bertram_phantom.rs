// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety mirror for the Bertram phantom burster
//! Four-state Bertram et al. (2000) phantom-burster equations 1–10.

const VOLTAGE_MIN: f64 = -250.0;
const VOLTAGE_MAX: f64 = 250.0;
const GATE_TOLERANCE: f64 = 1.0e-9;

/// Complete source state and parameter contract in mV, ms, pS, and fF.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BertramPhantomBurster {
    pub v: f64,
    pub n: f64,
    pub s1: f64,
    pub s2: f64,
    pub lambda_n: f64,
    pub g_ca: f64,
    pub g_k: f64,
    pub g_s1: f64,
    pub g_s2: f64,
    pub g_l: f64,
    pub e_ca: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub v_m: f64,
    pub s_m: f64,
    pub v_n: f64,
    pub s_n: f64,
    pub v_s1: f64,
    pub s_s1: f64,
    pub v_s2: f64,
    pub s_s2: f64,
    pub tau_n_bar: f64,
    pub tau_s1: f64,
    pub tau_s2: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl Default for BertramPhantomBurster {
    fn default() -> Self {
        Self::new()
    }
}

impl BertramPhantomBurster {
    /// Construct the Figure 2 author-code operating point.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            v: -43.0,
            n: 0.03,
            s1: 0.1,
            s2: 0.434,
            lambda_n: 1.1,
            g_ca: 280.0,
            g_k: 1300.0,
            g_s1: 20.0,
            g_s2: 32.0,
            g_l: 25.0,
            e_ca: 100.0,
            e_k: -80.0,
            e_l: -40.0,
            c_m: 4524.0,
            v_m: -22.0,
            s_m: 7.5,
            v_n: -9.0,
            s_n: 10.0,
            v_s1: -40.0,
            s_s1: 0.5,
            v_s2: -42.0,
            s_s2: 0.4,
            tau_n_bar: 9.09,
            tau_s1: 1000.0,
            tau_s2: 120_000.0,
            dt: 0.5,
            v_threshold: -20.0,
        }
    }

    fn boltz(v: f64, midpoint: f64, slope: f64) -> f64 {
        let z = (midpoint - v) / slope;
        if z >= 0.0 {
            let exp_negative = (-z).exp();
            exp_negative / (1.0 + exp_negative)
        } else {
            1.0 / (1.0 + z.exp())
        }
    }

    fn derivatives(&self, state: (f64, f64, f64, f64), current: f64) -> (f64, f64, f64, f64) {
        let (v, n, s1, s2) = state;
        let m_inf = Self::boltz(v, self.v_m, self.s_m);
        let n_inf = Self::boltz(v, self.v_n, self.s_n);
        let s1_inf = Self::boltz(v, self.v_s1, self.s_s1);
        let s2_inf = Self::boltz(v, self.v_s2, self.s_s2);
        let tau_n = self.tau_n_bar / (1.0 + ((v - self.v_n) / self.s_n).exp());

        let i_ca = self.g_ca * m_inf * (v - self.e_ca);
        let i_k = self.g_k * n * (v - self.e_k);
        let i_s1 = self.g_s1 * s1 * (v - self.e_k);
        let i_s2 = self.g_s2 * s2 * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        (
            (-i_ca - i_k - i_s1 - i_s2 - i_l + current) / self.c_m,
            self.lambda_n * (n_inf - n) / tau_n,
            (s1_inf - s1) / self.tau_s1,
            (s2_inf - s2) / self.tau_s2,
        )
    }

    fn add_scaled(
        state: (f64, f64, f64, f64),
        derivative: (f64, f64, f64, f64),
        scale: f64,
    ) -> (f64, f64, f64, f64) {
        (
            state.0 + scale * derivative.0,
            state.1 + scale * derivative.1,
            state.2 + scale * derivative.2,
            state.3 + scale * derivative.3,
        )
    }

    fn candidate(&self, current: f64) -> (f64, f64, f64, f64) {
        let state = (self.v, self.n, self.s1, self.s2);
        let k1 = self.derivatives(state, current);
        let k2 = self.derivatives(Self::add_scaled(state, k1, 0.5 * self.dt), current);
        let k3 = self.derivatives(Self::add_scaled(state, k2, 0.5 * self.dt), current);
        let k4 = self.derivatives(Self::add_scaled(state, k3, self.dt), current);
        let scale = self.dt / 6.0;
        (
            state.0 + scale * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0),
            state.1 + scale * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1),
            state.2 + scale * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2),
            state.3 + scale * (k1.3 + 2.0 * k2.3 + 2.0 * k3.3 + k4.3),
        )
    }

    /// Advance one simultaneous RK4 sample. Errors preserve all state.
    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() || !self.valid() {
            return Err("invalid Bertram phantom input");
        }
        let previous_v = self.v;
        let (v, n, s1, s2) = self.candidate(current);
        if !Self::candidate_valid(v, n, s1, s2) {
            return Err("invalid Bertram phantom candidate");
        }
        self.v = v;
        self.n = n.clamp(0.0, 1.0);
        self.s1 = s1.clamp(0.0, 1.0);
        self.s2 = s2.clamp(0.0, 1.0);
        Ok(i32::from(
            self.v >= self.v_threshold && previous_v < self.v_threshold,
        ))
    }

    /// Restore the source initial state while preserving configuration.
    pub fn reset(&mut self) {
        self.v = -43.0;
        self.n = 0.03;
        self.s1 = 0.1;
        self.s2 = 0.434;
    }

    fn candidate_valid(v: f64, n: f64, s1: f64, s2: f64) -> bool {
        v.is_finite()
            && (VOLTAGE_MIN..=VOLTAGE_MAX).contains(&v)
            && [n, s1, s2].iter().all(|value| {
                value.is_finite() && (-GATE_TOLERANCE..=1.0 + GATE_TOLERANCE).contains(value)
            })
    }

    /// Validate the complete source state and parameter domain.
    #[must_use]
    pub fn valid(&self) -> bool {
        Self::candidate_valid(self.v, self.n, self.s1, self.s2)
            && self.lambda_n.is_finite()
            && self.lambda_n > 0.0
            && [self.g_ca, self.g_k, self.g_s1, self.g_s2, self.g_l]
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
            && [
                self.e_ca, self.e_k, self.e_l, self.v_m, self.v_n, self.v_s1, self.v_s2,
            ]
            .iter()
            .all(|value| value.is_finite())
            && [
                self.c_m,
                self.s_m,
                self.s_n,
                self.s_s1,
                self.s_s2,
                self.tau_n_bar,
                self.tau_s1,
                self.tau_s2,
                self.dt,
            ]
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
            && self.v_threshold.is_finite()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_defaults_are_valid() {
        assert!(BertramPhantomBurster::new().valid());
    }

    #[test]
    fn dynamic_potassium_gate_moves() {
        let mut model = BertramPhantomBurster::new();
        assert!(model.step(0.0).is_ok());
        let expected = [
            -42.962_466_678_980_54,
            0.030_142_733_666_228_928,
            0.099_951_295_945_267_4,
            0.433_998_521_816_373_7,
        ];
        for (actual, reference) in [model.v, model.n, model.s1, model.s2]
            .into_iter()
            .zip(expected)
        {
            assert!((actual - reference).abs() < 5.0e-13);
        }
    }

    #[test]
    fn invalid_drive_is_atomic() {
        let mut model = BertramPhantomBurster::new();
        let before = model;
        assert!(model.step(f64::NAN).is_err());
        assert_eq!(model, before);
    }
}
