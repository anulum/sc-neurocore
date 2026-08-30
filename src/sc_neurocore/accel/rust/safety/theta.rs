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

pub type ThetaCompleteTrace = (Vec<f64>, Vec<u8>, f64);

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
        if !event_packet_representable(self, i_ext) {
            return Err("theta step can contain more than one source event");
        }

        let (next_theta, spiked) = self.exact_candidate(i_ext);
        if !next_theta.is_finite() {
            return Err("theta exact-flow update became non-finite");
        }

        self.theta = wrap_phase(next_theta);
        if spiked {
            Ok(1)
        } else {
            Ok(0)
        }
    }

    pub fn reset(&mut self) {
        self.theta = 0.0_f64;
    }

    pub fn simulate_complete(
        &self,
        n_steps: usize,
        i_ext: f64,
    ) -> Result<ThetaCompleteTrace, &'static str> {
        if !i_ext.is_finite() || !validate_theta(self) || !event_packet_representable(self, i_ext) {
            return Err("invalid theta batch contract");
        }
        let mut candidate = self.clone();
        let mut phase = Vec::with_capacity(n_steps);
        let mut events = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            let event = candidate.step(i_ext)?;
            phase.push(candidate.theta);
            events.push(event as u8);
        }
        Ok((phase, events, candidate.theta))
    }

    fn exact_candidate(&self, i_ext: f64) -> (f64, bool) {
        let y = (self.theta / 2.0).tan();
        if i_ext > 0.0 {
            let root_i = i_ext.sqrt();
            let phase = (y / root_i).atan();
            let next_phase = phase + root_i * self.dt;
            let spiked = next_phase >= std::f64::consts::FRAC_PI_2;
            if next_phase.cos().abs() <= 1.0e-15 {
                return (-std::f64::consts::PI, spiked);
            }
            return (wrap_phase(2.0 * (root_i * next_phase.tan()).atan()), spiked);
        }
        if i_ext == 0.0 {
            let denominator = 1.0 - y * self.dt;
            if denominator.abs() <= 1.0e-15 {
                return (-std::f64::consts::PI, true);
            }
            let next_y = y / denominator;
            return (wrap_phase(2.0 * next_y.atan()), denominator <= 0.0);
        }

        let root_i = (-i_ext).sqrt();
        if (y + root_i).abs() <= 1.0e-15 {
            return (self.theta, false);
        }
        let ratio = (y - root_i) / (y + root_i);
        let evolved = ratio * (2.0 * root_i * self.dt).exp();
        let denominator = 1.0 - evolved;
        let spiked = (ratio < 1.0 && evolved >= 1.0) || denominator.abs() <= 1.0e-15;
        if spiked && denominator.abs() <= 1.0e-15 {
            return (-std::f64::consts::PI, true);
        }
        let next_y = root_i * (1.0 + evolved) / denominator;
        (wrap_phase(2.0 * next_y.atan()), spiked)
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

pub fn event_packet_representable(state: &ThetaNeuron, i_ext: f64) -> bool {
    i_ext <= 0.0 || i_ext.sqrt() * state.dt <= std::f64::consts::PI
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
    fn test_exact_positive_flow_matches_closed_form() {
        let mut state = ThetaNeuron {
            theta: 1.0,
            dt: 0.2,
        };
        let current = 2.0_f64;
        let root_i = current.sqrt();
        let next_phase = ((state.theta / 2.0).tan() / root_i).atan() + root_i * state.dt;
        let expected = wrap_phase(2.0 * (root_i * next_phase.tan()).atan());
        let spike = state.step(current).unwrap();
        assert_eq!(spike, 0);
        assert!((state.theta - expected).abs() < 1.0e-12);
    }

    #[test]
    fn test_exact_flow_reports_within_step_crossing() {
        let mut state = ThetaNeuron {
            theta: 2.5,
            dt: 1.0,
        };
        assert_eq!(state.step(1.0).unwrap(), 1);
        assert!(state.theta >= -std::f64::consts::PI);
        assert!(state.theta <= std::f64::consts::PI);
    }

    #[test]
    fn test_negative_current_fixed_point_is_preserved() {
        let mut state = ThetaNeuron {
            theta: -std::f64::consts::FRAC_PI_2,
            dt: 100.0,
        };
        assert_eq!(state.step(-1.0).unwrap(), 0);
        assert!((state.theta + std::f64::consts::FRAC_PI_2).abs() < 1.0e-12);
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

    #[test]
    fn test_enrolled_event_vector_matches_python_reference() {
        let cases = [
            (-1.0, 0),
            (-0.5, 0),
            (0.0, 0),
            (0.1, 1),
            (0.333, 2),
            (0.5, 2),
            (1.0, 3),
            (2.0, 5),
            (5.0, 7),
            (20.0, 14),
            (50.0, 23),
        ];
        for (current, expected) in cases {
            let mut state = ThetaNeuron::new();
            let spikes: i32 = (0..1_000).map(|_| state.step(current).unwrap()).sum();
            assert_eq!(spikes, expected, "current={current}");
        }
    }

    #[test]
    fn test_complete_packet_and_multi_event_rejection_are_atomic() {
        let state = ThetaNeuron {
            theta: 0.37,
            dt: 0.037,
        };
        let (phase, events, final_theta) = state.simulate_complete(400, 2.2).unwrap();
        assert_eq!(phase.len(), 400);
        assert_eq!(events.len(), 400);
        assert_eq!(phase.last().copied(), Some(final_theta));
        assert_eq!(state.theta, 0.37);

        let mut rejected = ThetaNeuron {
            theta: 0.25,
            dt: 1.0,
        };
        assert!(rejected.step(16.0).is_err());
        assert_eq!(rejected.theta, 0.25);
    }
}
