// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for quadratic_if

#[derive(Debug, Clone)]
pub struct QuadraticIFNeuron {
    pub v: f64,
    pub v_reset: f64,
    pub v_peak: f64,
    pub dt: f64,
}

impl QuadraticIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -1.0_f64,
            v_reset: -1.0_f64,
            v_peak: 1.0_f64,
            dt: 0.01_f64,
        }
    }

    /// Construct Latham et al.'s normalized numerical source profile.
    pub fn latham_2000() -> Self {
        Self {
            v: -1.0,
            v_reset: -3.0,
            v_peak: 31.0 / 3.0,
            dt: 0.05,
        }
    }

    /// Return aligned voltage/events from a checked, failure-atomic clone.
    pub fn simulate_complete(
        &self,
        n_steps: usize,
        current: f64,
    ) -> Result<(Vec<f64>, Vec<u8>, f64), &'static str> {
        if !current.is_finite() || !validate_quadratic_if(self) {
            return Err("invalid quadratic-if batch contract");
        }
        let mut candidate = self.clone();
        let mut voltage = Vec::with_capacity(n_steps);
        let mut events = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            let event = candidate.step(current)?;
            voltage.push(candidate.v);
            events.push(event as u8);
        }
        Ok((voltage, events, candidate.v))
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !i_ext.is_finite() || !validate_quadratic_if(self) {
            return Err("quadratic-if state/current must be finite and well-formed");
        }

        let (next_v, spiked) = self.exact_candidate(i_ext);
        if !next_v.is_finite() {
            return Err("quadratic-if exact-flow update became non-finite");
        }

        self.v = next_v;
        if spiked {
            Ok(1)
        } else {
            Ok(0)
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_reset;
    }

    fn exact_candidate(&self, i_ext: f64) -> (f64, bool) {
        if i_ext > 0.0 {
            let root_i = i_ext.sqrt();
            let phase = (self.v / root_i).atan();
            let peak_phase = (self.v_peak / root_i).atan();
            let next_phase = phase + root_i * self.dt;
            if next_phase >= peak_phase || next_phase >= std::f64::consts::FRAC_PI_2 {
                return (self.v_reset, true);
            }
            return (root_i * next_phase.tan(), false);
        }
        if i_ext == 0.0 {
            let denominator = 1.0 - self.v * self.dt;
            if denominator <= 0.0 {
                return (self.v_reset, true);
            }
            let next_v = self.v / denominator;
            if next_v >= self.v_peak {
                return (self.v_reset, true);
            }
            return (next_v, false);
        }

        let root_i = (-i_ext).sqrt();
        if (self.v + root_i).abs() <= 1.0e-15 {
            return (self.v, false);
        }
        let numerator_ratio = (self.v - root_i) / (self.v + root_i);
        let evolved_ratio = numerator_ratio * (2.0 * root_i * self.dt).exp();
        let denominator = 1.0 - evolved_ratio;
        if (numerator_ratio < 1.0 && evolved_ratio >= 1.0) || denominator.abs() <= 1.0e-15 {
            return (self.v_reset, true);
        }
        let next_v = root_i * (1.0 + evolved_ratio) / denominator;
        if next_v >= self.v_peak {
            (self.v_reset, true)
        } else {
            (next_v, false)
        }
    }
}

impl Default for QuadraticIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_quadratic_if(state: &QuadraticIFNeuron) -> bool {
    state.v.is_finite()
        && state.v_reset.is_finite()
        && state.v_peak.is_finite()
        && state.dt.is_finite()
        && state.v < state.v_peak
        && state.v_reset < state.v_peak
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadratic_if_new() {
        let state = QuadraticIFNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_quadratic_if(&state));
    }

    #[test]
    fn test_quadratic_if_step() {
        let mut state = QuadraticIFNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_positive_current_spikes_and_resets() {
        let mut state = QuadraticIFNeuron::new();
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += state.step(1.0).unwrap();
        }
        assert!(spikes >= 100);
        assert!(state.v < state.v_peak);
    }

    #[test]
    fn test_invalid_current_does_not_mutate_state() {
        let mut state = QuadraticIFNeuron::new();
        state.v = -0.25;
        assert!(state.step(f64::NAN).is_err());
        assert_eq!(state.v, -0.25);
    }

    #[test]
    fn test_invalid_exact_flow_candidate_does_not_mutate_state() {
        let mut state = QuadraticIFNeuron::new();
        state.v = -0.25;
        assert!(state.step(-1.0e308).is_err());
        assert_eq!(state.v, -0.25);
    }

    #[test]
    fn test_exact_flow_matches_closed_form() {
        let mut state = QuadraticIFNeuron::new();
        let before = state.v;
        let current = 0.5_f64;
        let root_i = current.sqrt();
        let expected = root_i * ((before / root_i).atan() + root_i * state.dt).tan();
        let spike = state.step(current).unwrap();
        assert_eq!(spike, 0);
        assert!((state.v - expected).abs() < 1.0e-12);
    }

    #[test]
    fn test_reset_preserves_parameters() {
        let mut state = QuadraticIFNeuron {
            v: 0.5,
            v_reset: -2.0,
            v_peak: 2.0,
            dt: 0.005,
        };
        state.reset();
        assert_eq!(state.v, -2.0);
        assert_eq!(state.v_peak, 2.0);
        assert_eq!(state.dt, 0.005);
    }

    #[test]
    fn test_enrolled_event_vector_matches_python_reference() {
        let cases = [
            (0.0, 0),
            (0.333, 2),
            (0.5, 3),
            (1.0, 6),
            (2.0, 11),
            (5.0, 26),
            (20.0, 100),
            (50.0, 250),
        ];
        for (current, expected) in cases {
            let mut state = QuadraticIFNeuron::new();
            let spikes: i32 = (0..1_000).map(|_| state.step(current).unwrap()).sum();
            assert_eq!(spikes, expected, "current={current}");
        }
    }

    #[test]
    fn test_latham_source_complete_packet() {
        let state = QuadraticIFNeuron::latham_2000();
        let (voltage, events, final_v) = state.simulate_complete(240, 4.0).unwrap();
        assert_eq!(voltage.len(), 240);
        assert_eq!(events.len(), 240);
        assert_eq!(voltage.last().copied(), Some(final_v));
        assert_eq!(state.v, -1.0);
    }
}
