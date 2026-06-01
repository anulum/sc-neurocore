// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for gif_population

#[derive(Debug, Clone)]
pub struct GIFPopulationNeuron {
    pub v: f64,
    pub theta: f64,
    pub eta: f64,
    pub tau_m: f64,
    pub tau_eta: f64,
    pub delta_v: f64,
    pub lambda_0: f64,
    pub eta_increment: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub dt: f64,
    pub seed: u64,
    rng: u64,
}

impl GIFPopulationNeuron {
    pub fn new() -> Self {
        Self::with_seed(42)
    }

    pub fn with_seed(seed: u64) -> Self {
        let normalized_seed = if seed == 0 { 1 } else { seed };
        Self {
            v: -65.0,
            theta: -50.0,
            eta: 0.0,
            tau_m: 20.0,
            tau_eta: 100.0,
            delta_v: 2.0,
            lambda_0: 0.001,
            eta_increment: 5.0,
            v_rest: -65.0,
            v_reset: -65.0,
            dt: 0.5,
            seed: normalized_seed,
            rng: normalized_seed,
        }
    }

    fn finite_values(values: &[f64]) -> bool {
        values.iter().all(|value| value.is_finite())
    }

    fn valid_runtime(&self) -> bool {
        Self::finite_values(&[
            self.v,
            self.theta,
            self.eta,
            self.tau_m,
            self.tau_eta,
            self.delta_v,
            self.lambda_0,
            self.eta_increment,
            self.v_rest,
            self.v_reset,
            self.dt,
        ]) && self.tau_m > 0.0
            && self.tau_eta > 0.0
            && self.delta_v > 0.0
            && self.lambda_0 >= 0.0
            && self.dt > 0.0
    }

    fn uniform(&mut self) -> f64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        ((x.wrapping_mul(2685821657736338717) >> 11) as f64) * (1.0 / 9007199254740992.0)
    }

    fn advance_subthreshold(&self, i_ext: f64) -> Option<(f64, f64)> {
        let eta_decay = (-self.dt / self.tau_eta).exp();
        let membrane_decay = (-self.dt / self.tau_m).exp();
        let x0 = self.v - self.v_rest - i_ext;
        let eta_new = self.eta * eta_decay;
        let x_new = if (self.tau_m - self.tau_eta).abs() <= 1e-12 {
            membrane_decay * (x0 - self.eta * self.dt / self.tau_m)
        } else {
            let coupling = self.tau_eta / (self.tau_eta - self.tau_m);
            x0 * membrane_decay - self.eta * coupling * (eta_decay - membrane_decay)
        };
        let v_new = self.v_rest + i_ext + x_new;
        if Self::finite_values(&[v_new, eta_new]) {
            Some((v_new, eta_new))
        } else {
            None
        }
    }

    fn spike_probability(&self, voltage: f64) -> f64 {
        if self.lambda_0 == 0.0 {
            return 0.0;
        }
        let exponent = ((voltage - self.theta) / self.delta_v).clamp(-745.0, 20.0);
        let hazard = self.lambda_0 * exponent.exp();
        (1.0 - (-hazard * self.dt).exp()).clamp(0.0, 1.0)
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !i_ext.is_finite() || !self.valid_runtime() {
            return 0;
        }
        let Some((v_candidate, eta_candidate)) = self.advance_subthreshold(i_ext) else {
            return 0;
        };
        self.v = v_candidate;
        self.eta = eta_candidate;
        if self.uniform() < self.spike_probability(self.v) {
            self.v = self.v_reset;
            self.eta += self.eta_increment;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.eta = 0.0;
        self.rng = self.seed;
    }
}

pub fn validate_gif_population(state: &GIFPopulationNeuron) -> bool {
    state.valid_runtime()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gif_population_reference_point() {
        let mut state = GIFPopulationNeuron::with_seed(7);
        state.v = -68.0;
        state.eta = 0.4;
        assert_eq!(state.step(4.0), 0);
        assert!((state.v - (-67.8370206677805)).abs() < 1e-12);
        assert!((state.eta - 0.398004991677073).abs() < 1e-15);
    }

    #[test]
    fn test_gif_population_invalid_parameter_fails_closed() {
        let mut state = GIFPopulationNeuron::new();
        state.tau_m = 0.0;
        assert_eq!(state.step(1.0), 0);
        assert!(!validate_gif_population(&state));
    }

    #[test]
    fn test_gif_population_forced_spike() {
        let mut state = GIFPopulationNeuron::new();
        state.v = -51.0;
        state.eta = 0.3;
        state.theta = -90.0;
        state.lambda_0 = 1.0e9;
        assert_eq!(state.step(0.0), 1);
        assert!((state.v - state.v_reset).abs() < 1e-12);
        assert!((state.eta - 5.298503743757805).abs() < 1e-15);
    }
}
