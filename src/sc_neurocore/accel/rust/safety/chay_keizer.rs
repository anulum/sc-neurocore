// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety mirror for the Chay & Keizer 1983 burster

#![allow(dead_code)]

const MAX_SUBSTEP: f64 = 0.01;
const V_MIN: f64 = -200.0;
const V_MAX: f64 = 200.0;
const CA_MAX: f64 = 1000.0;

#[derive(Debug, Clone)]
pub struct ChayKeizerNeuron {
    pub v: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
    pub ca: f64,
    pub g_ca: f64,
    pub g_k: f64,
    pub g_kca: f64,
    pub g_l: f64,
    pub e_ca: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub v_prime: f64,
    pub v_star: f64,
    pub k_dis: f64,
    pub radius_cm: f64,
    pub faraday: f64,
    pub f_ca: f64,
    pub k_ca: f64,
    pub temp_celsius: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl ChayKeizerNeuron {
    pub fn new() -> Self {
        Self {
            v: -54.774,
            m: 0.029725,
            h: 0.747865,
            n: 0.061079,
            ca: 0.8,
            g_ca: 6.5,
            g_k: 12.0,
            g_kca: 0.09,
            g_l: 0.04,
            e_ca: 100.0,
            e_k: -75.0,
            e_l: -40.0,
            c_m: 1.0,
            v_prime: 50.0,
            v_star: 30.0,
            k_dis: 1.0,
            radius_cm: 8.9e-4,
            faraday: 96487.0,
            f_ca: 0.004,
            k_ca: 0.04,
            temp_celsius: 20.0,
            dt: 0.05,
            v_threshold: -30.0,
        }
    }

    fn checked_exp(exponent: f64) -> Result<f64, &'static str> {
        if !exponent.is_finite() {
            return Err("exponent must be finite");
        }
        Ok(exponent.clamp(-700.0, 700.0).exp())
    }

    fn alpha_m(&self, v: f64) -> Result<f64, &'static str> {
        let d = (v + self.v_prime) - 25.0;
        if d.abs() < 1e-7 {
            Ok(1.0)
        } else {
            Ok(-0.1 * d / (Self::checked_exp(-d / 10.0)? - 1.0))
        }
    }

    fn beta_m(&self, v: f64) -> Result<f64, &'static str> {
        Ok(4.0 * Self::checked_exp(-(v + self.v_prime) / 18.0)?)
    }

    fn alpha_h(&self, v: f64) -> Result<f64, &'static str> {
        Ok(0.07 * Self::checked_exp(-(v + self.v_prime) / 20.0)?)
    }

    fn beta_h(&self, v: f64) -> Result<f64, &'static str> {
        Ok(1.0 / (Self::checked_exp(-((v + self.v_prime) - 30.0) / 10.0)? + 1.0))
    }

    fn alpha_n(&self, v: f64) -> Result<f64, &'static str> {
        let d = (v + self.v_star) - 10.0;
        if d.abs() < 1e-7 {
            Ok(0.1)
        } else {
            Ok(-0.01 * d / (Self::checked_exp(-d / 10.0)? - 1.0))
        }
    }

    fn beta_n(&self, v: f64) -> Result<f64, &'static str> {
        Ok(0.125 * Self::checked_exp(-(v + self.v_star) / 80.0)?)
    }

    fn validate(&self) -> Result<(usize, f64, f64, f64), &'static str> {
        if !self.v.is_finite() || !(V_MIN..=V_MAX).contains(&self.v) {
            return Err("v outside Chay-Keizer safety envelope");
        }
        if [self.m, self.h, self.n]
            .iter()
            .any(|gate| !gate.is_finite() || !(0.0..=1.0).contains(gate))
        {
            return Err("Chay-Keizer gate must be in [0, 1]");
        }
        if !self.ca.is_finite() || !(0.0..=CA_MAX).contains(&self.ca) {
            return Err("ca outside Chay-Keizer safety envelope");
        }
        if [
            self.g_ca, self.g_k, self.g_kca, self.g_l, self.f_ca, self.k_ca,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err("non-negative Chay-Keizer parameter invalid");
        }
        if [
            self.e_ca,
            self.e_k,
            self.e_l,
            self.v_prime,
            self.v_star,
            self.temp_celsius,
            self.v_threshold,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err("finite Chay-Keizer parameter invalid");
        }
        if [self.c_m, self.k_dis, self.radius_cm, self.faraday, self.dt]
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err("positive Chay-Keizer parameter invalid");
        }

        let substeps = (self.dt / MAX_SUBSTEP).ceil().max(1.0) as usize;
        if substeps > 100_000 {
            return Err("dt requires too many Chay-Keizer safety substeps");
        }
        let phi = 3.0_f64.powf((self.temp_celsius - 6.3) / 10.0);
        let ca_influx = 3.0 / (self.radius_cm * self.faraday);
        if !phi.is_finite() || !ca_influx.is_finite() {
            return Err("derived Chay-Keizer parameter invalid");
        }
        Ok((substeps, self.dt / substeps as f64, phi, ca_influx))
    }

    #[allow(clippy::too_many_arguments)]
    fn candidate(
        &self,
        v: f64,
        m: f64,
        h: f64,
        n: f64,
        ca: f64,
        current: f64,
        step_dt: f64,
        phi: f64,
        ca_influx: f64,
    ) -> Result<(f64, f64, f64, f64, f64), &'static str> {
        let g_ca_open = self.g_ca * m * m * m * h;
        let i_ca = g_ca_open * (self.e_ca - v);
        let i_k = self.g_k * n * n * n * n * (self.e_k - v);
        let i_kca = self.g_kca * (ca / (ca + self.k_dis)) * (self.e_k - v);
        let i_l = self.g_l * (self.e_l - v);

        let v_next = v + (current + 2.0 * i_ca + i_k + i_kca + i_l) / self.c_m * step_dt;
        let m_next = m + phi * (self.alpha_m(v)? * (1.0 - m) - self.beta_m(v)? * m) * step_dt;
        let h_next = h + phi * (self.alpha_h(v)? * (1.0 - h) - self.beta_h(v)? * h) * step_dt;
        let n_next = n + phi * (self.alpha_n(v)? * (1.0 - n) - self.beta_n(v)? * n) * step_dt;
        let ca_next = ca + self.f_ca * (ca_influx * i_ca - self.k_ca * ca) * step_dt;

        if !v_next.is_finite() || !(V_MIN..=V_MAX).contains(&v_next) {
            return Err("Chay-Keizer voltage candidate outside safety envelope");
        }
        if !ca_next.is_finite() || !(0.0..=CA_MAX).contains(&ca_next) {
            return Err("Chay-Keizer calcium candidate outside safety envelope");
        }
        Ok((
            v_next,
            m_next.clamp(0.0, 1.0),
            h_next.clamp(0.0, 1.0),
            n_next.clamp(0.0, 1.0),
            ca_next,
        ))
    }

    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() {
            return Err("current must be finite");
        }
        let (substeps, step_dt, phi, ca_influx) = self.validate()?;
        let v_initial = self.v;
        let (mut v, mut m, mut h, mut n, mut ca) = (self.v, self.m, self.h, self.n, self.ca);
        let mut crossed = false;

        for _ in 0..substeps {
            let (v_next, m_next, h_next, n_next, ca_next) =
                self.candidate(v, m, h, n, ca, current, step_dt, phi, ca_influx)?;
            crossed |= v_next >= self.v_threshold && v < self.v_threshold;
            (v, m, h, n, ca) = (v_next, m_next, h_next, n_next, ca_next);
        }

        (self.v, self.m, self.h, self.n, self.ca) = (v, m, h, n, ca);
        Ok(i32::from(crossed && v_initial < self.v_threshold))
    }

    pub fn reset(&mut self) {
        (self.v, self.m, self.h, self.n, self.ca) = (-54.774, 0.029725, 0.747865, 0.061079, 0.8);
    }
}

pub fn validate_chay_keizer(state: &ChayKeizerNeuron) -> bool {
    state.validate().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_reference_after_500_steps() {
        let mut state = ChayKeizerNeuron::new();
        let spikes: i32 = (0..500).map(|_| state.step(5.0).unwrap()).sum();
        assert_eq!(spikes, 5);
        assert!((state.v - (-22.178_677_540_385_234)).abs() < 1e-10);
        assert!((state.m - 0.561_718_429_848_010_1).abs() < 1e-12);
        assert!((state.h - 0.050_707_724_198_745_974).abs() < 1e-12);
        assert!((state.n - 0.378_348_118_340_997_74).abs() < 1e-12);
        assert!((state.ca - 0.812_797_569_122_101_7).abs() < 1e-12);
        assert!(validate_chay_keizer(&state));
    }

    #[test]
    fn invalid_runtime_state_is_rejected_without_mutation() {
        let mut state = ChayKeizerNeuron::new();
        state.n = 1.5;
        let before = (state.v, state.m, state.h, state.n, state.ca);
        assert!(state.step(0.0).is_err());
        assert_eq!((state.v, state.m, state.h, state.n, state.ca), before);
    }

    #[test]
    fn candidate_failure_is_rejected_without_mutation() {
        let mut state = ChayKeizerNeuron::new();
        state.radius_cm = 1e-15;
        let before = (state.v, state.m, state.h, state.n, state.ca);
        assert!(state.step(0.0).is_err());
        assert_eq!((state.v, state.m, state.h, state.n, state.ca), before);
    }

    #[test]
    fn reset_restores_all_dynamic_state() {
        let mut state = ChayKeizerNeuron::new();
        for _ in 0..100 {
            state.step(5.0).unwrap();
        }
        state.reset();
        assert_eq!(
            (state.v, state.m, state.h, state.n, state.ca),
            (-54.774, 0.029725, 0.747865, 0.061079, 0.8)
        );
    }
}
