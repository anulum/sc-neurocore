// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for theta

#[derive(Debug, Clone)]
pub struct ThetaNeuron {
    pub theta: f64,
    pub dt: f64,
}

impl ThetaNeuron {
    pub fn new() -> Self {
        Self {
            theta: 0.0_f64,
            dt: 0.01_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !i_ext.is_finite() || !validate_theta(self) {
            return Err("theta state/current must be finite with positive dt");
        }

        let theta_prev = self.theta;
        let cos_theta = self.theta.cos();
        let dtheta = ((1.0 - cos_theta) + (1.0 + cos_theta) * i_ext) * self.dt;
        let next_theta = self.theta + dtheta;
        if !dtheta.is_finite() || !next_theta.is_finite() {
            return Err("theta phase increment became non-finite");
        }

        let spike = if theta_prev < std::f64::consts::PI * 0.99
            && next_theta >= std::f64::consts::PI * 0.99
        {
            1
        } else {
            0
        };
        self.theta = wrap_phase(next_theta);
        Ok(spike)
    }

    pub fn reset(&mut self) {
        self.theta = 0.0_f64;
    }
}

impl Default for ThetaNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_theta(state: &ThetaNeuron) -> bool {
    state.theta.is_finite() && state.dt.is_finite() && state.dt > 0.0
}

pub fn wrap_phase(theta: f64) -> f64 {
    (theta + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theta_new() {
        let state = ThetaNeuron::new();
        assert!(validate_theta(&state));
    }

    #[test]
    fn test_theta_step() {
        let mut state = ThetaNeuron::new();
        let spike = state.step(10.0).expect("valid step must succeed");
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_positive_current_spikes_and_wraps() {
        let mut state = ThetaNeuron::new();
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += state.step(1.0).unwrap();
        }
        assert!(spikes >= 100);
        assert!(state.theta >= -std::f64::consts::PI);
        assert!(state.theta <= std::f64::consts::PI);
    }

    #[test]
    fn test_invalid_current_does_not_mutate_state() {
        let mut state = ThetaNeuron::new();
        state.theta = 0.25;
        assert!(state.step(f64::NAN).is_err());
        assert_eq!(state.theta, 0.25);
    }

    #[test]
    fn test_invalid_phase_increment_does_not_mutate_state() {
        let mut state = ThetaNeuron::new();
        state.theta = 0.25;
        state.dt = 1.0e308;
        assert!(state.step(1.0e308).is_err());
        assert_eq!(state.theta, 0.25);
    }

    #[test]
    fn test_reset_preserves_dt() {
        let mut state = ThetaNeuron {
            theta: 2.0,
            dt: 0.005,
        };
        state.reset();
        assert_eq!(state.theta, 0.0);
        assert_eq!(state.dt, 0.005);
    }
}
