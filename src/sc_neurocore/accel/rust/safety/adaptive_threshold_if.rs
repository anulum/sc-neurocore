// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Standalone Rust safety mirror for adaptive-threshold IF

#[derive(Debug, Clone)]
pub struct AdaptiveThresholdIFNeuron {
    pub v: f64,
    pub theta: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub theta_rest: f64,
    pub delta_theta: f64,
    pub tau_m: f64,
    pub tau_theta: f64,
    pub dt: f64,
}

impl AdaptiveThresholdIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            theta: -50.0,
            v_rest: -65.0,
            v_reset: -65.0,
            theta_rest: -50.0,
            delta_theta: 5.0,
            tau_m: 10.0,
            tau_theta: 50.0,
            dt: 0.1,
        }
    }

    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() || !validate_adaptive_threshold_if(self) {
            return Err("adaptive-threshold state/current must be finite and well-formed");
        }
        let next_v = exact_relaxation(self.v, self.v_rest + current, self.tau_m, self.dt)?;
        let next_theta = exact_relaxation(self.theta, self.theta_rest, self.tau_theta, self.dt)?;
        if next_v >= next_theta {
            let spike_theta = next_theta + self.delta_theta;
            if !spike_theta.is_finite() {
                return Err("adaptive-threshold threshold jump became non-finite");
            }
            self.v = self.v_reset;
            self.theta = spike_theta;
            return Ok(1);
        }
        self.v = next_v;
        self.theta = next_theta;
        Ok(0)
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.theta = self.theta_rest;
    }
}

impl Default for AdaptiveThresholdIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_adaptive_threshold_if(state: &AdaptiveThresholdIFNeuron) -> bool {
    state.v.is_finite()
        && state.theta.is_finite()
        && state.v_rest.is_finite()
        && state.v_reset.is_finite()
        && state.theta_rest.is_finite()
        && state.theta_rest > state.v_rest
        && state.theta_rest > state.v_reset
        && state.delta_theta.is_finite()
        && state.delta_theta >= 0.0
        && state.tau_m.is_finite()
        && state.tau_m > 0.0
        && state.tau_theta.is_finite()
        && state.tau_theta > 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
}

fn exact_relaxation(state: f64, steady_state: f64, tau: f64, dt: f64) -> Result<f64, &'static str> {
    let decay = (-dt / tau).exp();
    let candidate = steady_state + (state - steady_state) * decay;
    if !candidate.is_finite() {
        return Err("adaptive-threshold exact-relaxation update became non-finite");
    }
    Ok(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_defaults_are_valid() {
        let state = AdaptiveThresholdIFNeuron::new();
        assert!(validate_adaptive_threshold_if(&state));
        assert_eq!(
            (state.v, state.theta, state.delta_theta, state.tau_m, state.tau_theta, state.dt),
            (-65.0, -50.0, 5.0, 10.0, 50.0, 0.1)
        );
    }

    #[test]
    fn exact_relaxation_without_spike() {
        let mut state = AdaptiveThresholdIFNeuron {
            v: -60.0,
            theta: -52.0,
            v_rest: -70.0,
            v_reset: -68.0,
            theta_rest: -48.0,
            delta_theta: 3.0,
            tau_m: 8.0,
            tau_theta: 40.0,
            dt: 0.05,
        };
        let decay_v = (-state.dt / state.tau_m).exp();
        let decay_theta = (-state.dt / state.tau_theta).exp();
        let expected_v = (state.v_rest + 12.5) + (state.v - (state.v_rest + 12.5)) * decay_v;
        let expected_theta = state.theta_rest + (state.theta - state.theta_rest) * decay_theta;
        assert_eq!(state.step(12.5).unwrap(), 0);
        assert!((state.v - expected_v).abs() < 1.0e-12);
        assert!((state.theta - expected_theta).abs() < 1.0e-12);
    }

    #[test]
    fn crossing_installs_reset_and_fixed_threshold_shift() {
        let mut state = AdaptiveThresholdIFNeuron {
            v: -50.5,
            theta: -51.0,
            v_rest: -65.0,
            v_reset: -65.0,
            theta_rest: -50.0,
            delta_theta: 5.0,
            tau_m: 10.0,
            tau_theta: 50.0,
            dt: 0.1,
        };
        assert_eq!(state.step(0.0).unwrap(), 1);
        assert_eq!(state.v, -65.0);
        let decay_theta = (-0.1_f64 / 50.0).exp();
        let relaxed = -50.0 + (-51.0 + 50.0) * decay_theta;
        assert!((state.theta - (relaxed + 5.0)).abs() < 1.0e-12);
        assert_eq!(state.step(0.0).unwrap(), 0);
    }

    #[test]
    fn below_threshold_stays_silent() {
        let mut state = AdaptiveThresholdIFNeuron::new();
        let mut spikes = 0;
        for _ in 0..500 {
            spikes += state.step(0.0).unwrap();
        }
        assert_eq!(spikes, 0);
    }

    #[test]
    fn invalid_current_does_not_mutate_state() {
        let mut state = AdaptiveThresholdIFNeuron::new();
        state.v = -60.0;
        state.theta = -55.0;
        let before = (state.v, state.theta);
        assert!(state.step(f64::NAN).is_err());
        assert_eq!((state.v, state.theta), before);
    }

    #[test]
    fn invalid_configuration_does_not_mutate_state() {
        let mut state = AdaptiveThresholdIFNeuron::new();
        state.v = -60.0;
        state.theta = -55.0;
        state.tau_m = -1.0;
        let before = (state.v, state.theta);
        assert!(state.step(1.0).is_err());
        assert_eq!((state.v, state.theta), before);
    }

    #[test]
    fn invalid_update_does_not_mutate_state() {
        let mut state = AdaptiveThresholdIFNeuron::new();
        state.v = -f64::MAX;
        let before = (state.v, state.theta);
        assert!(state.step(f64::MAX).is_err());
        assert_eq!((state.v, state.theta), before);
    }

    #[test]
    fn reset_preserves_configuration() {
        let mut state = AdaptiveThresholdIFNeuron {
            v: -55.0,
            theta: -40.0,
            v_rest: -70.0,
            v_reset: -68.0,
            theta_rest: -48.0,
            delta_theta: 3.0,
            tau_m: 8.0,
            tau_theta: 40.0,
            dt: 0.05,
        };
        state.reset();
        assert_eq!((state.v, state.theta), (-70.0, -48.0));
        assert_eq!(
            (state.v_rest, state.v_reset, state.theta_rest, state.delta_theta, state.tau_m, state.tau_theta, state.dt),
            (-70.0, -68.0, -48.0, 3.0, 8.0, 40.0, 0.05)
        );
    }
}
