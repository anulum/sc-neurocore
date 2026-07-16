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

    pub fn step(&mut self, current: f64) -> i32 {
        if !current.is_finite() || !self.valid() {
            return 0;
        }
        let (next_theta, spiked) = self.exact_candidate(current);
        if !next_theta.is_finite() {
            return 0;
        }
        self.theta = Self::wrap_phase(next_theta);
        if spiked {
            1
        } else {
            0
        }
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
        let mut n = ThetaNeuron::default();
        n.theta = 1.0;
        n.dt = 0.2;
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
        let mut n = ThetaNeuron::default();
        n.theta = 2.5;
        n.dt = 1.0;
        assert_eq!(n.step(1.0), 1);
        assert!(n.theta >= -std::f64::consts::PI && n.theta <= std::f64::consts::PI);
    }
    #[test]
    fn theta_stable_fixed_point_preserved() {
        let mut n = ThetaNeuron::default();
        n.theta = -std::f64::consts::FRAC_PI_2;
        n.dt = 100.0;
        assert_eq!(n.step(-1.0), 0);
        assert!((n.theta + std::f64::consts::FRAC_PI_2).abs() < 1.0e-12);
    }
    #[test]
    fn theta_non_finite_exact_candidate_preserves_state() {
        let mut n = ThetaNeuron::default();
        n.theta = 0.25;
        n.dt = 1.0e308;
        let before = n.theta;
        assert_eq!(n.step(-1.0e308), 0);
        assert_eq!(n.theta, before);
    }
}
