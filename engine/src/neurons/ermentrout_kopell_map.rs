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
/// theta(n+1) = theta(n) + dt * (1 - cos(theta)) + (1 + cos(theta)) * I
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

    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let theta_prev = self.theta;

        let d_theta = (1.0 - self.theta.cos()) + (1.0 + self.theta.cos()) * input;
        self.theta += self.dt * d_theta;

        // Spike detection: crossing pi
        let fired = if self.theta >= self.theta_threshold && theta_prev < self.theta_threshold {
            1
        } else {
            0
        };

        // Wrap theta to [0, 2*pi)
        let two_pi = 2.0 * std::f64::consts::PI;
        if self.theta >= two_pi {
            self.theta -= two_pi;
        }
        if self.theta < 0.0 {
            self.theta += two_pi;
        }

        if !self.theta.is_finite() {
            self.theta = 0.0;
        }

        fired
    }

    /// Run `n_steps` under a constant input, returning the `theta` trace
    /// (wrapped to `[0, 2*pi)`) and the upward-crossing spike count. Reuses
    /// `step` so the trace matches the per-step path; on a shared libm it also
    /// matches the Python reference bit-for-bit (the only transcendental is
    /// `cos`, and the non-chaotic phase flow does not amplify ULP differences).
    /// The final state is left in `self.theta`.
    pub fn simulate(&mut self, n_steps: usize, current: f64) -> (Vec<f64>, i64) {
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes: i64 = 0;
        for _ in 0..n_steps {
            let spiked = self.step(current);
            trace.push(self.theta);
            spikes += spiked as i64;
        }
        (trace, spikes)
    }

    pub fn reset(&mut self) {
        *self = Self::new();
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
    fn ek_nan_input_stays_finite() {
        let mut n = ErmentroutKopellMapNeuron::new();
        n.step(f64::NAN);
        assert!(n.theta.is_finite());
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
        for _ in 0..100 {
            n.step(0.5);
        }
        n.reset();
        assert_eq!(n.theta, 0.0);
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
