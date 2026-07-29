// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Chay & Keizer 1983 pancreatic beta-cell burster

//! Faithful five-state Chay-Keizer pancreatic beta-cell dynamics.

const MAX_SUBSTEP: f64 = 0.01;
const V_MIN: f64 = -200.0;
const V_MAX: f64 = 200.0;
const CA_MAX: f64 = 1000.0;

/// Chay & Keizer 1983 five-state pancreatic beta-cell model.
#[derive(Clone, Debug)]
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

    fn checked_exp(exponent: f64) -> f64 {
        exponent.clamp(-700.0, 700.0).exp()
    }

    fn alpha_m(&self, v: f64) -> f64 {
        let d = (v + self.v_prime) - 25.0;
        if d.abs() < 1e-7 {
            1.0
        } else {
            -0.1 * d / (Self::checked_exp(-d / 10.0) - 1.0)
        }
    }

    fn beta_m(&self, v: f64) -> f64 {
        4.0 * Self::checked_exp(-(v + self.v_prime) / 18.0)
    }

    fn alpha_h(&self, v: f64) -> f64 {
        0.07 * Self::checked_exp(-(v + self.v_prime) / 20.0)
    }

    fn beta_h(&self, v: f64) -> f64 {
        1.0 / (Self::checked_exp(-((v + self.v_prime) - 30.0) / 10.0) + 1.0)
    }

    fn alpha_n(&self, v: f64) -> f64 {
        let d = (v + self.v_star) - 10.0;
        if d.abs() < 1e-7 {
            0.1
        } else {
            -0.01 * d / (Self::checked_exp(-d / 10.0) - 1.0)
        }
    }

    fn beta_n(&self, v: f64) -> f64 {
        0.125 * Self::checked_exp(-(v + self.v_star) / 80.0)
    }

    fn valid_state(&self, current: f64) -> bool {
        current.is_finite()
            && self.v.is_finite()
            && (V_MIN..=V_MAX).contains(&self.v)
            && [self.m, self.h, self.n]
                .iter()
                .all(|gate| gate.is_finite() && (0.0..=1.0).contains(gate))
            && self.ca.is_finite()
            && (0.0..=CA_MAX).contains(&self.ca)
            && [
                self.g_ca, self.g_k, self.g_kca, self.g_l, self.f_ca, self.k_ca,
            ]
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
            && [
                self.e_ca,
                self.e_k,
                self.e_l,
                self.v_prime,
                self.v_star,
                self.temp_celsius,
                self.v_threshold,
            ]
            .iter()
            .all(|value| value.is_finite())
            && [self.c_m, self.k_dis, self.radius_cm, self.faraday, self.dt]
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            && (self.dt / MAX_SUBSTEP).ceil() <= 100_000.0
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
    ) -> Option<(f64, f64, f64, f64, f64)> {
        let g_ca_open = self.g_ca * m * m * m * h;
        let i_ca = g_ca_open * (self.e_ca - v);
        let i_k = self.g_k * n * n * n * n * (self.e_k - v);
        let i_kca = self.g_kca * (ca / (ca + self.k_dis)) * (self.e_k - v);
        let i_l = self.g_l * (self.e_l - v);

        let v_next = v + (current + 2.0 * i_ca + i_k + i_kca + i_l) / self.c_m * step_dt;
        let m_next = m + phi * (self.alpha_m(v) * (1.0 - m) - self.beta_m(v) * m) * step_dt;
        let h_next = h + phi * (self.alpha_h(v) * (1.0 - h) - self.beta_h(v) * h) * step_dt;
        let n_next = n + phi * (self.alpha_n(v) * (1.0 - n) - self.beta_n(v) * n) * step_dt;
        let ca_next = ca + self.f_ca * (ca_influx * i_ca - self.k_ca * ca) * step_dt;

        if !v_next.is_finite()
            || !(V_MIN..=V_MAX).contains(&v_next)
            || !ca_next.is_finite()
            || !(0.0..=CA_MAX).contains(&ca_next)
        {
            return None;
        }
        Some((
            v_next,
            m_next.clamp(0.0, 1.0),
            h_next.clamp(0.0, 1.0),
            n_next.clamp(0.0, 1.0),
            ca_next,
        ))
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !self.valid_state(current) {
            return 0;
        }

        let substeps = (self.dt / MAX_SUBSTEP).ceil().max(1.0) as usize;
        let step_dt = self.dt / substeps as f64;
        let phi = 3.0_f64.powf((self.temp_celsius - 6.3) / 10.0);
        let ca_influx = 3.0 / (self.radius_cm * self.faraday);
        let v_initial = self.v;
        let (mut v, mut m, mut h, mut n, mut ca) = (self.v, self.m, self.h, self.n, self.ca);
        let mut crossed = false;

        for _ in 0..substeps {
            let Some((v_next, m_next, h_next, n_next, ca_next)) =
                self.candidate(v, m, h, n, ca, current, step_dt, phi, ca_influx)
            else {
                return 0;
            };
            crossed |= v_next >= self.v_threshold && v < self.v_threshold;
            (v, m, h, n, ca) = (v_next, m_next, h_next, n_next, ca_next);
        }

        (self.v, self.m, self.h, self.n, self.ca) = (v, m, h, n, ca);
        i32::from(crossed && v_initial < self.v_threshold)
    }

    pub fn reset(&mut self) {
        (self.v, self.m, self.h, self.n, self.ca) = (-54.774, 0.029725, 0.747865, 0.061079, 0.8);
    }
}

impl Default for ChayKeizerNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_python_reference_after_500_steps() {
        let mut neuron = ChayKeizerNeuron::new();
        let spikes: i32 = (0..500).map(|_| neuron.step(5.0)).sum();
        assert_eq!(spikes, 5);
        assert!((neuron.v - (-22.178_677_540_385_234)).abs() < 1e-10);
        assert!((neuron.m - 0.561_718_429_848_010_1).abs() < 1e-12);
        assert!((neuron.h - 0.050_707_724_198_745_974).abs() < 1e-12);
        assert!((neuron.n - 0.378_348_118_340_997_74).abs() < 1e-12);
        assert!((neuron.ca - 0.812_797_569_122_101_7).abs() < 1e-12);
    }

    #[test]
    fn reset_restores_all_dynamic_state() {
        let mut neuron = ChayKeizerNeuron::new();
        for _ in 0..100 {
            neuron.step(5.0);
        }
        neuron.reset();
        assert_eq!(
            (neuron.v, neuron.m, neuron.h, neuron.n, neuron.ca),
            (-54.774, 0.029725, 0.747865, 0.061079, 0.8)
        );
    }

    #[test]
    fn invalid_input_does_not_mutate_state() {
        let mut neuron = ChayKeizerNeuron::new();
        let before = (neuron.v, neuron.m, neuron.h, neuron.n, neuron.ca);
        assert_eq!(neuron.step(f64::NAN), 0);
        assert_eq!((neuron.v, neuron.m, neuron.h, neuron.n, neuron.ca), before);
    }

    #[test]
    fn candidate_failure_does_not_partially_mutate_state() {
        let mut neuron = ChayKeizerNeuron::new();
        neuron.radius_cm = 1e-15;
        let before = (neuron.v, neuron.m, neuron.h, neuron.n, neuron.ca);
        assert_eq!(neuron.step(0.0), 0);
        assert_eq!((neuron.v, neuron.m, neuron.h, neuron.n, neuron.ca), before);
    }
}
