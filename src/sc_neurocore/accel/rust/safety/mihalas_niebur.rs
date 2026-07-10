// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for Mihalas-Niebur RK4 dynamics

#[derive(Debug, Clone)]
pub struct MihalasNieburNeuron {
    pub v: f64,
    pub theta: f64,
    pub i1: f64,
    pub i2: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub theta_reset: f64,
    pub theta_inf: f64,
    pub tau_v: f64,
    pub tau_theta: f64,
    pub tau_1: f64,
    pub tau_2: f64,
    pub a: f64,
    pub b: f64,
    pub r1: f64,
    pub r2: f64,
    pub dt: f64,
}

impl MihalasNieburNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0,
            theta: 1.0,
            i1: 0.0,
            i2: 0.0,
            v_rest: 0.0,
            v_reset: 0.0,
            theta_reset: 1.0,
            theta_inf: 1.0,
            tau_v: 10.0,
            tau_theta: 100.0,
            tau_1: 10.0,
            tau_2: 200.0,
            a: 0.0,
            b: 0.0,
            r1: 0.0,
            r2: 0.0,
            dt: 1.0,
        }
    }

    fn finite_values(values: &[f64]) -> bool {
        values.iter().all(|value| value.is_finite())
    }

    fn valid_runtime(&self) -> bool {
        Self::finite_values(&[
            self.v,
            self.theta,
            self.i1,
            self.i2,
            self.v_rest,
            self.v_reset,
            self.theta_reset,
            self.theta_inf,
            self.tau_v,
            self.tau_theta,
            self.tau_1,
            self.tau_2,
            self.a,
            self.b,
            self.r1,
            self.r2,
            self.dt,
        ]) && self.tau_v > 0.0
            && self.tau_theta > 0.0
            && self.tau_1 > 0.0
            && self.tau_2 > 0.0
            && self.dt > 0.0
    }

    fn derivatives(&self, v: f64, theta: f64, i1: f64, i2: f64, i_ext: f64) -> [f64; 4] {
        [
            (-(v - self.v_rest) + i1 + i2 + i_ext) / self.tau_v,
            (self.theta_inf - theta + self.a * (v - self.v_rest)) / self.tau_theta,
            -i1 / self.tau_1,
            -i2 / self.tau_2,
        ]
    }

    fn add_scaled(state: [f64; 4], slope: [f64; 4], scale: f64) -> [f64; 4] {
        [
            state[0] + scale * slope[0],
            state[1] + scale * slope[1],
            state[2] + scale * slope[2],
            state[3] + scale * slope[3],
        ]
    }

    fn rk4_candidate(&self, i_ext: f64) -> Option<[f64; 4]> {
        let state = [self.v, self.theta, self.i1, self.i2];
        let half_dt = 0.5 * self.dt;
        let k1 = self.derivatives(state[0], state[1], state[2], state[3], i_ext);
        let s2 = Self::add_scaled(state, k1, half_dt);
        let k2 = self.derivatives(s2[0], s2[1], s2[2], s2[3], i_ext);
        let s3 = Self::add_scaled(state, k2, half_dt);
        let k3 = self.derivatives(s3[0], s3[1], s3[2], s3[3], i_ext);
        let s4 = Self::add_scaled(state, k3, self.dt);
        let k4 = self.derivatives(s4[0], s4[1], s4[2], s4[3], i_ext);
        let candidate = [
            state[0] + self.dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state[1] + self.dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            state[2] + self.dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
            state[3] + self.dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
        ];
        if Self::finite_values(&candidate) {
            Some(candidate)
        } else {
            None
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !i_ext.is_finite() || !self.valid_runtime() {
            return 0;
        }
        let Some(candidate) = self.rk4_candidate(i_ext) else {
            return 0;
        };
        self.v = candidate[0];
        self.theta = candidate[1];
        self.i1 = candidate[2];
        self.i2 = candidate[3];
        if self.v >= self.theta {
            self.v = self.v_reset + self.b * (self.v - self.v_rest);
            self.theta = self.theta.max(self.theta_reset);
            self.i1 += self.r1;
            self.i2 += self.r2;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.theta = self.theta_reset;
        self.i1 = 0.0;
        self.i2 = 0.0;
    }
}

impl Default for MihalasNieburNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_mihalas_niebur(state: &MihalasNieburNeuron) -> bool {
    state.valid_runtime()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mihalas_niebur_new() {
        let state = MihalasNieburNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_mihalas_niebur(&state));
    }

    #[test]
    fn test_mihalas_niebur_rk4_reference_point() {
        let mut state = MihalasNieburNeuron::new();
        assert_eq!(state.step(0.5), 0);
        assert!((state.v - 0.04758125).abs() < 1e-12);
        assert!((state.theta - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_mihalas_niebur_spike_reset_uses_b() {
        let mut state = MihalasNieburNeuron::new();
        state.v = 0.99;
        state.b = 0.5;
        state.r1 = 1.25;
        state.r2 = -0.25;
        assert_eq!(state.step(2.0), 1);
        assert!((state.v - 0.5430570625).abs() < 1e-12);
        assert!((state.i1 - 1.25).abs() < 1e-15);
        assert!((state.i2 - (-0.25)).abs() < 1e-15);
    }

    #[test]
    fn test_mihalas_niebur_invalid_input_preserves_state() {
        let mut state = MihalasNieburNeuron::new();
        state.v = 0.2;
        state.i1 = 0.3;
        let before = (state.v, state.theta, state.i1, state.i2);
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!((state.v, state.theta, state.i1, state.i2), before);
    }

    #[test]
    fn matches_python_golden_spike_count() {
        // Parity with models/mihalas_niebur.py (RK4 integrator, default parameters). The
        // Mihalas-Niebur 2009 right-hand side is entirely linear (leaky membrane, linearly-adapting
        // threshold, two exponentially-decaying internal currents), with a spike (v >= theta) that
        // resets v to v_reset + b*(v - v_rest), lifts theta to at least theta_reset, and increments
        // the internal currents. No transcendentals, so the trajectory is bit-for-bit across
        // languages and the spike count is an exact observable. At the default parameters the
        // adaptation coefficient and after-spike increments are zero, so this exercises the
        // sustained-firing regime of a fixed-threshold reset (the adaptation and b-scaled reset are
        // covered exactly by test_mihalas_niebur_spike_reset_uses_b). Drive gates it cleanly around
        // rheobase (~1): silent at I=0.0, a 142-spike train at I=2.0, a 333-spike train at I=5.0,
        // each over 1000 macro steps. Verified python-vs-rust max|Δ|=0; the Go, Julia, Mojo and
        // Rust-engine backends reproduce the same trajectory via test_mihalas_niebur_backends.py.
        for (current, want) in [(0.0_f64, 0_usize), (2.0, 142), (5.0, 333)] {
            let mut state = MihalasNieburNeuron::new();
            let spikes = (0..1000).filter(|_| state.step(current) == 1).count();
            assert_eq!(spikes, want, "I={current}");
        }
    }
}
