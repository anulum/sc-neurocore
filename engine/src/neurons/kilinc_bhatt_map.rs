// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Kilinc-Bhatt adaptive-threshold map neuron

//! Kilinc-Bhatt adaptive-threshold map neuron.

/// Kilinc-Bhatt 2023 — sigmoid map with adaptive threshold.
///
/// Minimal 2D map with built-in spike frequency adaptation via
/// a slow threshold variable. Designed for efficient hardware
/// implementation while retaining biologically relevant dynamics.
///
/// x(n+1) = k * sigmoid(x(n) - theta(n)) + I
/// theta(n+1) = beta * theta(n) + gamma * H(x(n) - theta_spike)
///
/// H() is the Heaviside step function (spike-triggered increment).
#[derive(Clone, Debug)]
pub struct KilincBhattMapNeuron {
    pub x: f64,
    pub theta: f64,       // Adaptive threshold
    pub k: f64,           // Gain
    pub beta: f64,        // Threshold decay
    pub gamma: f64,       // Spike→threshold coupling
    pub theta_spike: f64, // Spike detection level
    pub x_threshold: f64,
}

impl Default for KilincBhattMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl KilincBhattMapNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            theta: 0.0,
            k: 1.5,
            beta: 0.95,
            gamma: 0.3,
            theta_spike: 0.8,
            x_threshold: 0.8,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let x_prev = self.x;
        let sig = 1.0 / (1.0 + (-(self.x - self.theta) * 4.0).exp());
        let x_new = -self.x + self.k * sig + current;
        let spiked = if self.x >= self.theta_spike { 1.0 } else { 0.0 };
        let theta_new = self.beta * self.theta + self.gamma * spiked;

        self.x = x_new.clamp(-5.0, 5.0);
        self.theta = theta_new.clamp(-5.0, 5.0);

        if !self.x.is_finite() {
            self.x = 0.0;
        }
        if !self.theta.is_finite() {
            self.theta = 0.0;
        }

        if self.x >= self.x_threshold && x_prev < self.x_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kb_fires_with_input() {
        let mut n = KilincBhattMapNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(1.0)).sum();
        assert!(t > 0, "KB must fire with input, got {t}");
    }

    #[test]
    fn kb_silent_without_input() {
        let mut n = KilincBhattMapNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0, "KB must be silent without input, got {t}");
    }

    #[test]
    fn kb_adaptation() {
        // Theta increases with spiking → fewer spikes over time
        let mut n = KilincBhattMapNeuron::new();
        let early: i32 = (0..2000).map(|_| n.step(1.0)).sum();
        let late: i32 = (0..2000).map(|_| n.step(1.0)).sum();
        assert!(
            early >= late,
            "Adaptation should slow firing: early={early}, late={late}"
        );
    }

    #[test]
    fn kb_theta_increases_during_spiking() {
        let mut n = KilincBhattMapNeuron::new();
        let theta_before = n.theta;
        for _ in 0..5000 {
            n.step(1.5);
        }
        assert!(
            n.theta > theta_before,
            "Theta must increase during spiking, theta={}",
            n.theta
        );
    }

    #[test]
    fn kb_negative_input_no_crash() {
        let mut n = KilincBhattMapNeuron::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.x.is_finite());
    }

    #[test]
    fn kb_nan_input_stays_finite() {
        let mut n = KilincBhattMapNeuron::new();
        n.step(f64::NAN);
        assert!(n.x.is_finite());
    }

    #[test]
    fn kb_extreme_input_bounded() {
        let mut n = KilincBhattMapNeuron::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.x.is_finite() && n.x <= 5.0);
    }

    #[test]
    fn kb_reset_clears_state() {
        let mut n = KilincBhattMapNeuron::new();
        for _ in 0..100 {
            n.step(1.0);
        }
        n.reset();
        assert_eq!(n.x, 0.0);
        assert_eq!(n.theta, 0.0);
    }

    #[test]
    fn kb_performance_100k_steps() {
        let start = std::time::Instant::now();
        let mut n = KilincBhattMapNeuron::new();
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
