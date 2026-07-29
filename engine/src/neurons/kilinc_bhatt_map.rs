// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Kilinc-Bhatt adaptive-threshold map neuron

//! Kilinc-Bhatt adaptive-threshold map neuron.

/// Experimental Nagumo-Sato/Aihara-derived sigmoid map with adaptive threshold.
///
/// Minimal 2D map with built-in spike frequency adaptation via
/// a slow threshold variable. Designed for efficient hardware
/// implementation while retaining biologically relevant dynamics.
///
/// x(n+1) = -x(n) + k * sigmoid(4 * (x(n) - theta(n))) + I
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

    fn valid(&self) -> bool {
        self.x.is_finite()
            && (-5.0..=5.0).contains(&self.x)
            && self.theta.is_finite()
            && (-5.0..=5.0).contains(&self.theta)
            && self.k.is_finite()
            && (0.0..=5.0).contains(&self.k)
            && self.beta.is_finite()
            && (0.0..=1.0).contains(&self.beta)
            && self.gamma.is_finite()
            && (0.0..=2.0).contains(&self.gamma)
            && self.theta_spike.is_finite()
            && (0.0..=2.0).contains(&self.theta_spike)
            && self.x_threshold.is_finite()
            && (0.0..=2.0).contains(&self.x_threshold)
    }

    fn sigmoid(z: f64) -> f64 {
        if z >= 0.0 {
            1.0 / (1.0 + (-z).exp())
        } else {
            let exp_z = z.exp();
            exp_z / (1.0 + exp_z)
        }
    }

    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() {
            return Err("current must be finite");
        }
        if !self.valid() {
            return Err("Kilinc-Bhatt state and parameters must satisfy the public bounds");
        }

        let x_prev = self.x;
        let sig = Self::sigmoid((self.x - self.theta) * 4.0);
        let x_new = -self.x + self.k * sig + current;
        let spiked = if self.x >= self.theta_spike { 1.0 } else { 0.0 };
        let theta_new = self.beta * self.theta + self.gamma * spiked;

        if !x_new.is_finite() || !theta_new.is_finite() {
            return Err("Kilinc-Bhatt candidate state became non-finite");
        }

        self.x = x_new.clamp(-5.0, 5.0);
        self.theta = theta_new.clamp(-5.0, 5.0);

        Ok(if self.x >= self.x_threshold && x_prev < self.x_threshold {
            1
        } else {
            0
        })
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    pub fn reset(&mut self) {
        self.x = 0.0;
        self.theta = 0.0;
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
    fn kb_nan_input_is_rejected_atomically() {
        let mut n = KilincBhattMapNeuron::new();
        let before = (n.x, n.theta);
        assert_eq!(n.try_step(f64::NAN), Err("current must be finite"));
        assert_eq!((n.x, n.theta), before);
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
        n.k = 2.0;
        n.beta = 0.8;
        for _ in 0..100 {
            n.step(1.0);
        }
        n.reset();
        assert_eq!(n.x, 0.0);
        assert_eq!(n.theta, 0.0);
        assert_eq!(n.k, 2.0);
        assert_eq!(n.beta, 0.8);
    }

    #[test]
    fn kb_invalid_state_is_rejected_atomically() {
        let mut n = KilincBhattMapNeuron::new();
        n.theta = f64::INFINITY;
        let before = (n.x, n.theta);
        assert!(n.try_step(1.0).is_err());
        assert_eq!((n.x, n.theta), before);
    }

    #[test]
    fn kb_sigmoid_is_stable_at_finite_extremes() {
        assert_eq!(KilincBhattMapNeuron::sigmoid(f64::MAX), 1.0);
        assert_eq!(KilincBhattMapNeuron::sigmoid(-f64::MAX), 0.0);
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
