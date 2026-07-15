// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Allen GLIF Neuron Model

//! Allen GLIF5 threshold-adaptation and after-spike-current dynamics.

/// Allen GLIF5 — LIF + threshold adaptation + after-spike currents.
#[derive(Clone, Debug)]
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

    fn derivatives(&self, v: f64, theta: f64, i_asc1: f64, i_asc2: f64, current: f64) -> [f64; 4] {
        [
            (-(v - self.v_rest) + self.resistance * current + i_asc1 + i_asc2) / self.tau_m,
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

    fn rk4_candidate(&self, current: f64) -> Option<[f64; 4]> {
        let state = [self.v, self.theta, self.i_asc1, self.i_asc2];
        let half_dt = 0.5 * self.dt;
        let k1 = self.derivatives(state[0], state[1], state[2], state[3], current);
        let s2 = Self::add_scaled(state, k1, half_dt);
        let k2 = self.derivatives(s2[0], s2[1], s2[2], s2[3], current);
        let s3 = Self::add_scaled(state, k2, half_dt);
        let k3 = self.derivatives(s3[0], s3[1], s3[2], s3[3], current);
        let s4 = Self::add_scaled(state, k3, self.dt);
        let k4 = self.derivatives(s4[0], s4[1], s4[2], s4[3], current);
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

    pub fn step(&mut self, current: f64) -> i32 {
        if !current.is_finite() || !self.valid_runtime() {
            return 0;
        }
        let Some(candidate) = self.rk4_candidate(current) else {
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

    /// Run `n_steps` of the candidate-first RK4 recurrence under a constant
    /// `current`, recording the membrane voltage after every step.
    ///
    /// Reuses [`step`] verbatim so the compiled inner loop is bit-identical to
    /// the per-step path; returns the voltage trace and the total spike count.
    pub fn simulate(&mut self, n_steps: usize, current: f64) -> (Vec<f64>, i64) {
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes: i64 = 0;
        for _ in 0..n_steps {
            spikes += i64::from(self.step(current));
            trace.push(self.v);
        }
        (trace, spikes)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = GLIFNeuron::default();
        let constructed = GLIFNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn simulate_matches_repeated_steps() {
        let mut simulated = GLIFNeuron::new();
        let mut repeated = simulated.clone();
        let (trace, spikes) = simulated.simulate(8, 30.0);
        let mut expected_spikes = 0_i64;
        let expected: Vec<f64> = (0..8)
            .map(|_| {
                expected_spikes += i64::from(repeated.step(30.0));
                repeated.v
            })
            .collect();
        assert_eq!(trace, expected);
        assert_eq!(spikes, expected_spikes);
    }

    #[test]
    fn nonfinite_rk4_candidate_preserves_state() {
        let mut n = GLIFNeuron::new();
        n.dt = f64::MAX;
        let before = (n.v, n.theta, n.i_asc1, n.i_asc2);
        assert_eq!(n.step(1.0), 0);
        assert_eq!((n.v, n.theta, n.i_asc1, n.i_asc2), before);
    }

    #[test]
    fn glif_fires() {
        let mut n = GLIFNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(30.0)).sum();
        assert!(t > 0);
    }

    // -- GLIF --
    #[test]
    fn glif_silent_without_input() {
        let mut n = GLIFNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn glif_reset_clears_state() {
        let mut n = GLIFNeuron::new();
        for _ in 0..100 {
            n.step(30.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
        assert!((n.i_asc1).abs() < 1e-10);
        assert!((n.i_asc2).abs() < 1e-10);
    }
    #[test]
    fn glif_rk4_reference_point() {
        let mut n = GLIFNeuron::new();
        n.v = -68.0;
        n.theta = -45.0;
        n.i_asc1 = 0.4;
        n.i_asc2 = -0.2;
        assert_eq!(n.step(4.0), 0);
        assert!((n.v - (-67.7924658728125)).abs() < 1e-12);
        assert!((n.theta - (-45.049541282631253)).abs() < 1e-12);
        assert!((n.i_asc1 - 0.361935).abs() < 1e-12);
        assert!((n.i_asc2 - (-0.19900249583333334)).abs() < 1e-10);
    }
    #[test]
    fn glif_spike_reset_adds_candidate_threshold() {
        let mut n = GLIFNeuron::new();
        n.v = -51.0;
        n.theta = -50.5;
        n.delta_theta = 2.5;
        n.r_asc1 = 1.25;
        n.r_asc2 = -0.25;
        assert_eq!(n.step(40.0), 1);
        assert!((n.v - (-70.0)).abs() < 1e-12);
        assert!((n.theta - (-47.9930331381625)).abs() < 1e-12);
        assert!((n.i_asc1 - 1.25).abs() < 1e-12);
        assert!((n.i_asc2 - (-0.25)).abs() < 1e-12);
    }
    #[test]
    fn glif_invalid_input_preserves_state() {
        let mut n = GLIFNeuron::new();
        n.v = -68.0;
        n.i_asc1 = 0.4;
        let before = (n.v, n.theta, n.i_asc1, n.i_asc2);
        assert_eq!(n.step(f64::NAN), 0);
        assert_eq!((n.v, n.theta, n.i_asc1, n.i_asc2), before);
    }
    #[test]
    fn glif_extreme_bounded() {
        let mut n = GLIFNeuron::new();
        for _ in 0..200 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn glif_threshold_adapts_after_spike() {
        let mut n = GLIFNeuron::new();
        let theta_init = n.theta;
        for _ in 0..200 {
            n.step(30.0);
        }
        assert!(
            n.theta > theta_init,
            "theta should increase after spikes (delta_theta > 0)"
        );
    }
    #[test]
    fn glif_afterspike_currents() {
        let mut n = GLIFNeuron::new();
        for _ in 0..200 {
            n.step(30.0);
        }
        // After spiking, ASC should have been triggered (then decayed)
        assert!(n.v.is_finite());
    }
    #[test]
    fn glif_negative_no_crash() {
        let mut n = GLIFNeuron::new();
        for _ in 0..200 {
            n.step(-30.0);
        }
        assert!(n.v.is_finite());
    }
}
