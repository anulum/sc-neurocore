// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for lapicque

#[derive(Debug, Clone)]
pub struct LapicqueNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau: f64,
    pub resistance: f64,
    pub dt: f64,
    pub capacitance: f64,
    pub series_resistance: f64,
    pub polarization_resistance: f64,
    pub excited: bool,
    pub source_profile: bool,
}

impl LapicqueNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0_f64,
            v_rest: 0.0_f64,
            v_reset: 0.0_f64,
            v_threshold: 1.0_f64,
            tau: 20.0_f64,
            resistance: 1.0_f64,
            dt: 1.0_f64,
            capacitance: 1.1_f64,
            series_resistance: 10.0_f64,
            polarization_resistance: 1.0_f64,
            excited: false,
            source_profile: false,
        }
    }

    /// Construct the normalized one-shot Lapicque 1907 polarization profile.
    pub fn lapicque_1907() -> Self {
        let mut state = Self::new();
        state.dt = 0.01;
        state.source_profile = true;
        state
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !i_ext.is_finite() {
            return Err("lapicque input current must be finite");
        }
        if !validate_lapicque(self) {
            return Err("lapicque state must satisfy finite positive-RC threshold contract");
        }

        let (v_inf, decay) = if self.source_profile {
            let total_resistance = self.series_resistance + self.polarization_resistance;
            let beta = self.capacitance
                * self.series_resistance
                * self.polarization_resistance
                / total_resistance;
            (
                i_ext * self.polarization_resistance / total_resistance,
                (-self.dt / beta).exp(),
            )
        } else {
            (
                self.v_rest + self.resistance * i_ext,
                (-self.dt / self.tau).exp(),
            )
        };
        let next_v = v_inf + (self.v - v_inf) * decay;
        if !v_inf.is_finite() || !decay.is_finite() || !next_v.is_finite() {
            return Err("lapicque voltage candidate must remain finite");
        }

        if self.source_profile {
            let event = !self.excited && next_v >= self.v_threshold;
            self.v = next_v;
            if event {
                self.excited = true;
                return Ok(1);
            }
            return Ok(0);
        }

        self.v = next_v;
        if next_v >= self.v_threshold {
            self.v = self.v_reset;
            Ok(1)
        } else {
            Ok(0)
        }
    }

    pub fn reset(&mut self) {
        self.v = if self.source_profile { 0.0 } else { self.v_rest };
        self.excited = false;
    }
}

impl Default for LapicqueNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_lapicque(state: &LapicqueNeuron) -> bool {
    let common = state.v.is_finite()
        && state.v_threshold.is_finite()
        && state.v_threshold > 0.0
        && state.dt.is_finite()
        && state.dt > 0.0;
    if !common {
        return false;
    }
    if state.source_profile {
        return (state.excited || state.v < state.v_threshold)
            && state.capacitance.is_finite()
            && state.capacitance > 0.0
            && state.series_resistance.is_finite()
            && state.series_resistance > 0.0
            && state.polarization_resistance.is_finite()
            && state.polarization_resistance > 0.0;
    }
    !state.excited
        && state.v_rest.is_finite()
        && state.v_reset.is_finite()
        && state.v_threshold > state.v_rest
        && state.v_threshold > state.v_reset
        && state.v < state.v_threshold
        && state.tau.is_finite()
        && state.tau > 0.0
        && state.resistance.is_finite()
        && state.resistance > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lapicque_new() {
        let state = LapicqueNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_lapicque(&state));
    }

    #[test]
    fn test_lapicque_step() {
        let mut state = LapicqueNeuron::new();
        let spike = state.step(10.0).expect("valid step must succeed");
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_lapicque_exact_flow_matches_closed_form() {
        let mut state = LapicqueNeuron::new();
        state.v = 0.25;
        state.dt = 5.0;
        let current = 0.5;
        let v0 = state.v;
        let v_inf = state.v_rest + state.resistance * current;
        let euler = v0 + (-(v0 - state.v_rest) + state.resistance * current) / state.tau * state.dt;
        let expected = v_inf + (v0 - v_inf) * (-state.dt / state.tau).exp();

        let spike = state.step(current).expect("valid step must succeed");
        assert_eq!(spike, 0);
        assert!((state.v - expected).abs() < 1e-15);
        assert!(
            (state.v - euler).abs() > 1e-4,
            "exact-flow candidate collapsed to Euler"
        );
    }

    #[test]
    fn test_positive_current_spikes_and_resets() {
        let mut state = LapicqueNeuron::new();
        let mut spikes = 0;
        for _ in 0..5_000 {
            spikes += state.step(20.0).expect("valid step must succeed");
        }
        assert!(spikes >= 100);
        assert!(state.v < state.v_threshold);
    }

    #[test]
    fn test_invalid_current_does_not_mutate_state() {
        let mut state = LapicqueNeuron::new();
        state.v = 0.25;
        assert!(state.step(f64::NAN).is_err());
        assert_eq!(state.v, 0.25);
    }

    #[test]
    fn test_invalid_runtime_state_does_not_mutate_state() {
        let mut state = LapicqueNeuron::new();
        state.v = 0.25;
        state.tau = 0.0;
        assert!(state.step(1.0).is_err());
        assert_eq!(state.v, 0.25);
    }

    #[test]
    fn test_invalid_increment_does_not_mutate_state() {
        let mut state = LapicqueNeuron::new();
        state.v = 0.25;
        state.v_threshold = 1.0e308;
        state.resistance = 1.0e308;
        assert!(state.step(1.0e308).is_err());
        assert_eq!(state.v, 0.25);
    }

    #[test]
    fn test_reset_preserves_parameters() {
        let mut state = LapicqueNeuron {
            v: 0.5,
            v_rest: -0.25,
            v_reset: -0.5,
            v_threshold: 2.0,
            tau: 10.0,
            resistance: 2.0,
            dt: 0.25,
            capacitance: 1.1,
            series_resistance: 10.0,
            polarization_resistance: 1.0,
            excited: false,
            source_profile: false,
        };
        state.reset();
        assert_eq!(state.v, -0.25);
        assert_eq!(state.v_reset, -0.5);
        assert_eq!(state.v_threshold, 2.0);
        assert_eq!(state.tau, 10.0);
        assert_eq!(state.resistance, 2.0);
        assert_eq!(state.dt, 0.25);
    }

    #[test]
    fn test_source_profile_matches_equation_and_latches_once() {
        let mut state = LapicqueNeuron::lapicque_1907();
        let mut events = 0;
        for _ in 0..200 {
            events += state.step(22.0).expect("source step must succeed");
        }
        assert_eq!(events, 1);
        assert!(state.excited);
        assert!(state.v > state.v_threshold);
        state.reset();
        assert_eq!((state.v, state.excited), (0.0, false));
    }
}
