// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Theta Neuron

/// Theta neuron — Ermentrout & Kopell canonical form.
/// dθ/dt = (1 - cosθ) + (1 + cosθ)·I, spike at θ crossing π.
#[derive(Clone, Debug)]
pub struct ThetaNeuron {
    pub theta: f64,
    pub dt: f64,
}

pub type ThetaCompleteTrace = (Vec<f64>, Vec<u8>, f64);

impl ThetaNeuron {
    pub fn new(dt: f64) -> Self {
        Self { theta: 0.0, dt }
    }

    fn wrap_phase(theta: f64) -> f64 {
        (theta + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
    }

    fn valid(&self) -> bool {
        self.theta.is_finite() && self.dt.is_finite() && self.dt > 0.0
    }

    fn event_packet_representable(&self, current: f64) -> bool {
        current <= 0.0 || current.sqrt() * self.dt <= std::f64::consts::PI
    }

    fn exact_candidate(&self, current: f64) -> (f64, bool) {
        let y = (self.theta / 2.0).tan();
        if current > 0.0 {
            let root_i = current.sqrt();
            let phase = (y / root_i).atan();
            let next_phase = phase + root_i * self.dt;
            if next_phase.cos().abs() <= 1.0e-15 {
                return (
                    -std::f64::consts::PI,
                    next_phase >= std::f64::consts::FRAC_PI_2,
                );
            }
            return (
                Self::wrap_phase(2.0 * (root_i * next_phase.tan()).atan()),
                next_phase >= std::f64::consts::FRAC_PI_2,
            );
        }
        if current == 0.0 {
            let denominator = 1.0 - y * self.dt;
            if denominator.abs() <= 1.0e-15 {
                return (-std::f64::consts::PI, true);
            }
            return (
                Self::wrap_phase(2.0 * (y / denominator).atan()),
                denominator <= 0.0,
            );
        }

        let root_i = (-current).sqrt();
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
        (
            Self::wrap_phase(2.0 * (root_i * (1.0 + evolved) / denominator).atan()),
            spiked,
        )
    }

    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() || !self.valid() {
            return Err("theta state/current must be finite with positive dt");
        }
        if !self.event_packet_representable(current) {
            return Err("theta step can contain more than one source event");
        }
        let (next_theta, spiked) = self.exact_candidate(current);
        if !next_theta.is_finite() {
            return Err("theta exact-flow candidate became non-finite");
        }
        self.theta = Self::wrap_phase(next_theta);
        Ok(i32::from(spiked))
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Execute a checked complete batch without mutating the source state.
    pub fn simulate_complete(
        &self,
        n_steps: usize,
        current: f64,
    ) -> Result<ThetaCompleteTrace, &'static str> {
        if !self.valid() || !current.is_finite() || !self.event_packet_representable(current) {
            return Err("invalid theta batch contract");
        }
        let mut candidate = self.clone();
        let mut phase = Vec::with_capacity(n_steps);
        let mut events = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            let event = candidate.try_step(current)?;
            phase.push(candidate.theta);
            events.push(event as u8);
        }
        Ok((phase, events, candidate.theta))
    }

    pub fn reset(&mut self) {
        self.theta = 0.0;
    }
}

impl Default for ThetaNeuron {
    fn default() -> Self {
        Self::new(0.01)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn theta_fires() {
        let mut n = ThetaNeuron::default();
        let total: i32 = (0..1000).map(|_| n.step(0.5)).sum();
        assert!(total > 0);
    }
    #[test]
    fn theta_silent_without_input() {
        let mut n = ThetaNeuron::default();
        let t: i32 = (0..1000).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn theta_reset_clears_state() {
        let mut n = ThetaNeuron::default();
        for _ in 0..100 {
            n.step(0.5);
        }
        n.reset();
        assert!((n.theta - 0.0).abs() < 1e-10);
    }
    #[test]
    fn theta_bounded() {
        let mut n = ThetaNeuron::default();
        for _ in 0..1000 {
            n.step(10.0);
        }
        assert!(n.theta.is_finite());
    }
    #[test]
    fn theta_nan_no_panic() {
        ThetaNeuron::default().step(f64::NAN);
    }
    #[test]
    fn theta_exact_positive_flow() {
        let mut n = ThetaNeuron {
            theta: 1.0,
            dt: 0.2,
        };
        let root_i = 2.0_f64.sqrt();
        let phase = ((n.theta / 2.0).tan() / root_i).atan();
        let expected =
            ThetaNeuron::wrap_phase(2.0 * (root_i * (phase + root_i * n.dt).tan()).atan());
        let spike = n.step(2.0);
        assert_eq!(spike, 0);
        assert!((n.theta - expected).abs() < 1.0e-12);
    }
    #[test]
    fn theta_exact_flow_reports_within_step_crossing() {
        let mut n = ThetaNeuron {
            theta: 2.5,
            dt: 1.0,
        };
        assert_eq!(n.step(1.0), 1);
        assert!(n.theta >= -std::f64::consts::PI && n.theta <= std::f64::consts::PI);
    }
    #[test]
    fn theta_stable_fixed_point_preserved() {
        let mut n = ThetaNeuron {
            theta: -std::f64::consts::FRAC_PI_2,
            dt: 100.0,
        };
        assert_eq!(n.step(-1.0), 0);
        assert!((n.theta + std::f64::consts::FRAC_PI_2).abs() < 1.0e-12);
    }
    #[test]
    fn theta_non_finite_exact_candidate_preserves_state() {
        let mut n = ThetaNeuron {
            theta: 0.25,
            dt: 1.0e308,
        };
        let before = n.theta;
        assert_eq!(n.step(-1.0e308), 0);
        assert_eq!(n.theta, before);
    }
    #[test]
    fn theta_complete_batch_is_aligned_and_failure_atomic() {
        let n = ThetaNeuron {
            theta: 0.37,
            dt: 0.037,
        };
        let (phase, events, final_theta) = n.simulate_complete(400, 2.2).unwrap();
        assert_eq!(phase.len(), 400);
        assert_eq!(events.len(), 400);
        assert_eq!(phase.last().copied(), Some(final_theta));
        assert_eq!(n.theta, 0.37);
        assert!(n.simulate_complete(1, f64::NAN).is_err());
        assert_eq!(n.theta, 0.37);
    }
    #[test]
    fn theta_rejects_multi_event_step_without_mutation() {
        let mut n = ThetaNeuron {
            theta: 0.25,
            dt: 1.0,
        };
        let before = n.theta;
        assert!(n.try_step(16.0).is_err());
        assert_eq!(n.theta, before);
    }
}
