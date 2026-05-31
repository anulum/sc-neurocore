// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for chay

#![allow(dead_code)]

const MAX_SUBSTEP: f64 = 0.001;
const V_MIN: f64 = -200.0;
const V_MAX: f64 = 200.0;
const CA_MAX: f64 = 100.0;

#[derive(Debug, Clone)]
pub struct ChayNeuron {
    pub v: f64,
    pub n: f64,
    pub ca: f64,
    pub g_ca: f64,
    pub g_k: f64,
    pub g_kca: f64,
    pub g_l: f64,
    pub e_ca: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub rho: f64,
    pub alpha_ca: f64,
    pub k_ca: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl ChayNeuron {
    pub fn new() -> Self {
        Self {
            v: -50.0,
            n: 0.1,
            ca: 0.1,
            g_ca: 25.0,
            g_k: 1400.0,
            g_kca: 12.0,
            g_l: 7.0,
            e_ca: 100.0,
            e_k: -75.0,
            e_l: -40.0,
            rho: 0.00015,
            alpha_ca: 0.002,
            k_ca: 0.04,
            dt: 0.02,
            v_threshold: -20.0,
        }
    }

    fn finite(value: f64) -> bool {
        value.is_finite()
    }

    fn valid_probability(value: f64) -> bool {
        value.is_finite() && (0.0..=1.0).contains(&value)
    }

    fn valid_nonnegative(value: f64) -> bool {
        value.is_finite() && value >= 0.0
    }

    fn checked_exp(exponent: f64) -> Result<f64, &'static str> {
        if !exponent.is_finite() {
            return Err("exponent must be finite");
        }
        if exponent < -700.0 {
            Ok(0.0)
        } else if exponent > 700.0 {
            Ok(700.0_f64.exp())
        } else {
            Ok(exponent.exp())
        }
    }

    fn gate_inf(exponent: f64) -> Result<f64, &'static str> {
        Ok(1.0 / (1.0 + Self::checked_exp(exponent)?))
    }

    fn validate(&self) -> Result<(usize, f64), &'static str> {
        if !Self::finite(self.v) || !(V_MIN..=V_MAX).contains(&self.v) {
            return Err("v outside Chay safety envelope");
        }
        if !Self::valid_probability(self.n) {
            return Err("n must be in [0, 1]");
        }
        if !Self::valid_nonnegative(self.ca) || self.ca > CA_MAX {
            return Err("ca outside Chay safety envelope");
        }
        for value in [
            self.g_ca,
            self.g_k,
            self.g_kca,
            self.g_l,
            self.rho,
            self.alpha_ca,
            self.k_ca,
        ] {
            if !Self::valid_nonnegative(value) {
                return Err("non-negative Chay parameter invalid");
            }
        }
        for value in [self.e_ca, self.e_k, self.e_l, self.v_threshold] {
            if !Self::finite(value) {
                return Err("finite Chay parameter invalid");
            }
        }
        if !self.dt.is_finite() || self.dt <= 0.0 {
            return Err("dt must be positive");
        }
        let substeps = (self.dt / MAX_SUBSTEP).ceil().max(1.0) as usize;
        if substeps > 10000 {
            return Err("dt requires too many Chay safety substeps");
        }
        Ok((substeps, self.dt / substeps as f64))
    }

    fn candidate(
        &self,
        v: f64,
        n: f64,
        ca: f64,
        h: f64,
        i_ext: f64,
    ) -> Result<(f64, f64, f64), &'static str> {
        let m_inf = Self::gate_inf(-(v + 25.0) / 8.0)?;
        let n_inf = Self::gate_inf(-(v + 18.0) / 14.0)?;
        let tau_n = 1.0 / (0.01 * (v + 18.0).abs().max(0.01));
        let ca_denominator = ca + 1.0;
        if ca_denominator <= 0.0 {
            return Err("calcium activation denominator must be positive");
        }

        let i_ca = self.g_ca * m_inf * (v - self.e_ca);
        let kca_act = ca / ca_denominator;
        let i_k = self.g_k * n * (v - self.e_k);
        let i_kca = self.g_kca * kca_act * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);

        let v_next = v + (-i_ca - i_k - i_kca - i_l + i_ext) * h;
        let n_next = n + (n_inf - n) / tau_n.max(0.01) * h;
        let ca_next = ca + self.rho * (-self.alpha_ca * i_ca - self.k_ca * ca) * h;

        if !v_next.is_finite() || !(V_MIN..=V_MAX).contains(&v_next) {
            return Err("Chay voltage candidate outside safety envelope");
        }
        if !Self::valid_probability(n_next) {
            return Err("Chay n-gate candidate outside [0, 1]");
        }
        if !Self::valid_nonnegative(ca_next) || ca_next > CA_MAX {
            return Err("Chay calcium candidate outside safety envelope");
        }
        Ok((v_next, n_next, ca_next))
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !i_ext.is_finite() {
            return Err("current must be finite");
        }
        let (substeps, h) = self.validate()?;
        let v_initial = self.v;
        let mut v = self.v;
        let mut n = self.n;
        let mut ca = self.ca;
        let mut crossed = false;

        for _ in 0..substeps {
            let (v_next, n_next, ca_next) = self.candidate(v, n, ca, h, i_ext)?;
            crossed = crossed || (v_next >= self.v_threshold && v < self.v_threshold);
            v = v_next;
            n = n_next;
            ca = ca_next;
        }

        self.v = v;
        self.n = n;
        self.ca = ca;
        Ok(if crossed && v_initial < self.v_threshold {
            1
        } else {
            0
        })
    }

    pub fn reset(&mut self) {
        self.v = -50.0;
        self.n = 0.1;
        self.ca = 0.1;
    }
}

pub fn validate_chay(state: &ChayNeuron) -> bool {
    state.validate().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_preserves_biophysical_bounds() {
        let mut state = ChayNeuron::new();
        for _ in 0..200 {
            let spike = state.step(0.0).unwrap();
            assert!(spike == 0 || spike == 1);
        }
        assert!(validate_chay(&state));
    }

    #[test]
    fn invalid_runtime_state_is_rejected_without_mutation() {
        let mut state = ChayNeuron::new();
        state.n = 1.5;
        let before = (state.v, state.n, state.ca);
        assert!(state.step(0.0).is_err());
        assert_eq!((state.v, state.n, state.ca), before);
    }

    #[test]
    fn current_depolarizes_relative_to_rest() {
        let mut rest = ChayNeuron::new();
        let mut driven = ChayNeuron::new();
        for _ in 0..50 {
            rest.step(0.0).unwrap();
            driven.step(1000.0).unwrap();
        }
        assert!(driven.v > rest.v);
    }
}
