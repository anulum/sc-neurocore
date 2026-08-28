// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Ermentrout-Kopell theta map neuron

//! Ermentrout-Kopell theta map neuron.

/// Ermentrout-Kopell canonical Type I — theta neuron in map form.
///
/// The canonical model for Type I (saddle-node) excitability.
/// theta(n+1) = theta(n) + dt * [(1 - cos(theta)) + (1 + cos(theta)) * gain * I]
/// Spike when theta crosses pi.
///
/// Ermentrout & Kopell, SIAM J Appl Math 46:233, 1986.
#[derive(Clone, Debug)]
pub struct ErmentroutKopellMapNeuron {
    pub theta: f64, // Phase variable [0, 2*pi)
    pub dt: f64,
    pub gain: f64,
    pub theta_threshold: f64,
}

impl Default for ErmentroutKopellMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl ErmentroutKopellMapNeuron {
    pub fn new() -> Self {
        Self {
            theta: 0.0,
            dt: 0.1, // Discrete step size
            gain: 1.0,
            theta_threshold: std::f64::consts::PI,
        }
    }

    fn parameters_are_valid(&self) -> bool {
        self.theta.is_finite()
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.gain.is_finite()
            && self.theta_threshold.is_finite()
    }

    /// Checked update. Rejected input or candidates leave phase unchanged.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !self.parameters_are_valid() {
            return Err("invalid Ermentrout-Kopell runtime state");
        }
        if !current.is_finite() {
            return Err("invalid Ermentrout-Kopell current");
        }
        let input = self.gain * current;
        if !input.is_finite() {
            return Err("invalid Ermentrout-Kopell input drive");
        }
        let theta_prev = self.theta;
        let cos_theta = theta_prev.cos();
        let d_theta = (1.0 - cos_theta) + (1.0 + cos_theta) * input;
        let theta_next = theta_prev + self.dt * d_theta;
        if !d_theta.is_finite() || !theta_next.is_finite() {
            return Err("invalid Ermentrout-Kopell candidate phase");
        }

        // Spike detection: crossing pi
        let fired = if theta_next >= self.theta_threshold && theta_prev < self.theta_threshold {
            1
        } else {
            0
        };

        self.theta = theta_next.rem_euclid(2.0 * std::f64::consts::PI);
        Ok(fired)
    }

    /// Infallible NetworkRunner adapter; invalid updates are event-silent and atomic.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Run `n_steps` under a constant input, returning the `theta` trace
    /// (wrapped to `[0, 2*pi)`) and the upward-crossing spike count. Reuses
    /// `step` so the trace matches the per-step path; on a shared libm it also
    /// matches the Python reference bit-for-bit (the only transcendental is
    /// `cos`, and the non-chaotic phase flow does not amplify ULP differences).
    /// The final state is left in `self.theta`.
    pub fn simulate(
        &mut self,
        n_steps: usize,
        current: f64,
    ) -> Result<(Vec<f64>, i64), &'static str> {
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes: i64 = 0;
        for _ in 0..n_steps {
            let spiked = self.try_step(current)?;
            trace.push(self.theta);
            spikes += spiked as i64;
        }
        Ok((trace, spikes))
    }

    pub fn reset(&mut self) {
        self.theta = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ek_fires_with_input() {
        let mut n = ErmentroutKopellMapNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(0.5)).sum();
        assert!(t > 0, "EK must fire with input, got {t}");
    }

    #[test]
    fn ek_silent_without_input() {
        // Type I: no firing below threshold (I < 0 is subthreshold for theta model)
        let mut n = ErmentroutKopellMapNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(-0.1)).sum();
        assert_eq!(t, 0, "EK must be silent with negative input, got {t}");
    }

    #[test]
    fn ek_type_i_excitability() {
        // Type I: arbitrarily low firing rate near threshold
        let mut n_low = ErmentroutKopellMapNeuron::new();
        let mut n_high = ErmentroutKopellMapNeuron::new();
        let spikes_low: i32 = (0..10_000).map(|_| n_low.step(0.01)).sum();
        let spikes_high: i32 = (0..10_000).map(|_| n_high.step(1.0)).sum();
        assert!(
            spikes_high > spikes_low,
            "Higher input → higher rate: high={spikes_high} vs low={spikes_low}"
        );
    }

    #[test]
    fn ek_theta_wraps() {
        // Theta should stay in [0, 2*pi)
        let mut n = ErmentroutKopellMapNeuron::new();
        for _ in 0..10_000 {
            n.step(0.5);
        }
        let two_pi = 2.0 * std::f64::consts::PI;
        assert!(
            n.theta >= 0.0 && n.theta < two_pi,
            "Theta must wrap to [0, 2pi), theta={}",
            n.theta
        );
    }

    #[test]
    fn ek_negative_input_no_crash() {
        let mut n = ErmentroutKopellMapNeuron::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.theta.is_finite());
    }

    #[test]
    fn ek_nan_input_is_event_silent_and_atomic() {
        let mut n = ErmentroutKopellMapNeuron::new();
        let before = n.theta;
        assert!(n.try_step(f64::NAN).is_err());
        assert_eq!(n.step(f64::NAN), 0);
        assert_eq!(n.theta, before);
    }

    #[test]
    fn ek_extreme_input_bounded() {
        let mut n = ErmentroutKopellMapNeuron::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.theta.is_finite());
    }

    #[test]
    fn ek_reset_clears_state() {
        let mut n = ErmentroutKopellMapNeuron::new();
        n.dt = 0.05;
        n.gain = 1.5;
        n.theta_threshold = 2.75;
        for _ in 0..100 {
            n.step(0.5);
        }
        n.reset();
        assert_eq!(n.theta, 0.0);
        assert_eq!((n.dt, n.gain, n.theta_threshold), (0.05, 1.5, 2.75));
    }

    #[test]
    fn ek_uses_true_circular_modulo_for_large_finite_steps() {
        let mut n = ErmentroutKopellMapNeuron::new();
        n.try_step(1.0e6).unwrap();
        assert!((0.0..2.0 * std::f64::consts::PI).contains(&n.theta));
    }

    #[test]
    fn ek_checked_simulation_returns_complete_trace() {
        let mut n = ErmentroutKopellMapNeuron::new();
        let (trace, events) = n.simulate(2_000, 0.5).unwrap();
        assert_eq!(trace.len(), 2_000);
        assert_eq!(events, 45);
        assert_eq!(trace.last().copied(), Some(n.theta));
    }

    #[test]
    fn ek_performance_100k_steps() {
        let start = std::time::Instant::now();
        let mut n = ErmentroutKopellMapNeuron::new();
        for _ in 0..100_000 {
            std::hint::black_box(n.step(0.5));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "100k steps must complete in <50ms"
        );
    }
}
