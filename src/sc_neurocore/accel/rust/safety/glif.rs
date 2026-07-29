// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for GLIF RK4 dynamics

#[derive(Debug, Clone)]
pub struct GLIFNeuron {
    pub v: f64,
    pub theta: f64,
    pub theta_inf: f64,
    pub i_asc1: f64,
    pub i_asc2: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub tau_m: f64,
    pub tau_theta: f64,
    pub tau_asc1: f64,
    pub tau_asc2: f64,
    pub a_theta: f64,
    pub delta_theta: f64,
    pub r_asc1: f64,
    pub r_asc2: f64,
    pub resistance: f64,
    pub dt: f64,
}

impl GLIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            theta: -50.0,
            theta_inf: -50.0,
            i_asc1: 0.0,
            i_asc2: 0.0,
            v_rest: -70.0,
            v_reset: -70.0,
            tau_m: 10.0,
            tau_theta: 100.0,
            tau_asc1: 10.0,
            tau_asc2: 200.0,
            a_theta: 0.01,
            delta_theta: 2.0,
            r_asc1: 1.0,
            r_asc2: 0.5,
            resistance: 1.0,
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
            self.theta_inf,
            self.i_asc1,
            self.i_asc2,
            self.v_rest,
            self.v_reset,
            self.tau_m,
            self.tau_theta,
            self.tau_asc1,
            self.tau_asc2,
            self.a_theta,
            self.delta_theta,
            self.r_asc1,
            self.r_asc2,
            self.resistance,
            self.dt,
        ]) && self.tau_m > 0.0
            && self.tau_theta > 0.0
            && self.tau_asc1 > 0.0
            && self.tau_asc2 > 0.0
            && self.dt > 0.0
            && self.delta_theta >= 0.0
            && self.resistance >= 0.0
    }

    fn derivatives(&self, v: f64, theta: f64, i_asc1: f64, i_asc2: f64, i_ext: f64) -> [f64; 4] {
        [
            (-(v - self.v_rest) + self.resistance * i_ext + i_asc1 + i_asc2) / self.tau_m,
            (self.theta_inf - theta + self.a_theta * (v - self.v_rest)) / self.tau_theta,
            -i_asc1 / self.tau_asc1,
            -i_asc2 / self.tau_asc2,
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
        let state = [self.v, self.theta, self.i_asc1, self.i_asc2];
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
        self.i_asc1 = candidate[2];
        self.i_asc2 = candidate[3];
        if self.v >= self.theta {
            self.v = self.v_reset;
            self.theta += self.delta_theta;
            self.i_asc1 += self.r_asc1;
            self.i_asc2 += self.r_asc2;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.theta = self.theta_inf;
        self.i_asc1 = 0.0;
        self.i_asc2 = 0.0;
    }
}

impl Default for GLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_glif(state: &GLIFNeuron) -> bool {
    state.valid_runtime()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glif_new() {
        let state = GLIFNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_glif(&state));
    }

    #[test]
    fn test_glif_rk4_reference_point() {
        let mut state = GLIFNeuron::new();
        state.v = -68.0;
        state.theta = -45.0;
        state.i_asc1 = 0.4;
        state.i_asc2 = -0.2;
        assert_eq!(state.step(4.0), 0);
        assert!((state.v - (-67.7924658728125)).abs() < 1e-12);
        assert!((state.theta - (-45.049_541_282_631_25)).abs() < 1e-12);
    }

    #[test]
    fn test_glif_spike_reset_adds_candidate_threshold() {
        let mut state = GLIFNeuron::new();
        state.v = -51.0;
        state.theta = -50.5;
        state.delta_theta = 2.5;
        state.r_asc1 = 1.25;
        state.r_asc2 = -0.25;
        assert_eq!(state.step(40.0), 1);
        assert!((state.v - (-70.0)).abs() < 1e-12);
        assert!((state.theta - (-47.9930331381625)).abs() < 1e-12);
        assert!((state.i_asc1 - 1.25).abs() < 1e-12);
        assert!((state.i_asc2 - (-0.25)).abs() < 1e-12);
    }

    #[test]
    fn test_glif_invalid_input_preserves_state() {
        let mut state = GLIFNeuron::new();
        state.v = -68.0;
        state.i_asc1 = 0.4;
        let before = (state.v, state.theta, state.i_asc1, state.i_asc2);
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!((state.v, state.theta, state.i_asc1, state.i_asc2), before);
    }

    #[test]
    fn matches_python_golden_spike_count() {
        // Parity with models/glif.py (RK4 integrator, default parameters). The GLIF (generalised
        // leaky integrate-and-fire) right-hand side is entirely linear — a leaky membrane, a
        // linearly-adapting threshold, and two exponentially-decaying after-spike currents — with
        // an adaptive-threshold spike (v >= theta) that hard-resets v to v_reset and increments
        // theta / the after-spike currents. No transcendentals, so the trajectory is bit-for-bit
        // across languages and the spike count is an exact observable. Drive gates the regime
        // cleanly around rheobase (~20-25): silent at I=0.0, a 54-spike regular train at I=30.0,
        // a 95-spike train at I=50.0, each over 1000 macro steps. Verified python-vs-rust
        // max|Δ|=0; the Go, Julia, Mojo and Rust-engine backends reproduce the same trajectory via
        // test_glif_backends.py.
        for (current, want) in [(0.0_f64, 0_usize), (30.0, 54), (50.0, 95)] {
            let mut state = GLIFNeuron::new();
            let spikes = (0..1000).filter(|_| state.step(current) == 1).count();
            assert_eq!(spikes, want, "I={current}");
        }
    }
}
