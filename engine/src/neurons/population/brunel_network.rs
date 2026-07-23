// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Brunel Balanced Network Model

/// Brunel 2000 — balanced excitatory/inhibitory network mean-field.
///
/// Two coupled rate equations for excitatory (r_e) and inhibitory (r_i)
/// populations. The balance of E and I determines the dynamical regime:
/// - Asynchronous Irregular (AI): low rates, Poisson-like
/// - Synchronous Regular (SR): oscillatory, gamma-band
/// - Synchronous Irregular (SI): fast oscillations, irregular single units
///
/// tau_e * dr_e/dt = -r_e + phi(J_ee*r_e - J_ei*r_i + I_ext)
/// tau_i * dr_i/dt = -r_i + phi(J_ie*r_e - J_ii*r_i)
///
/// phi() is a threshold-linear transfer function.
///
/// Brunel, J Comput Neurosci 8:183, 2000.
#[derive(Clone, Debug)]
pub struct BrunelNetwork {
    pub r_e: f64,       // Excitatory rate (Hz)
    pub r_i: f64,       // Inhibitory rate (Hz)
    pub tau_e: f64,     // Excitatory time constant (ms)
    pub tau_i: f64,     // Inhibitory time constant (ms)
    pub j_ee: f64,      // E→E coupling
    pub j_ei: f64,      // I→E coupling (inhibitory, positive value)
    pub j_ie: f64,      // E→I coupling
    pub j_ii: f64,      // I→I coupling (inhibitory, positive value)
    pub threshold: f64, // Transfer function threshold
    pub gain_phi: f64,  // Transfer function gain
    pub dt: f64,
    pub r_threshold: f64, // Spike detection threshold
    pub gain: f64,
}

impl Default for BrunelNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl BrunelNetwork {
    pub fn new() -> Self {
        Self {
            r_e: 0.1,
            r_i: 0.1,
            tau_e: 20.0,
            tau_i: 10.0,
            j_ee: 0.2,
            j_ei: 0.8, // Strong I→E inhibition
            j_ie: 0.5,
            j_ii: 0.2,
            threshold: 0.0,
            gain_phi: 1.0,
            dt: 0.1,
            r_threshold: 1.0,
            gain: 1.0,
        }
    }

    /// Threshold-linear transfer function
    fn phi(&self, x: f64) -> f64 {
        if x > self.threshold {
            self.gain_phi * (x - self.threshold)
        } else {
            0.0
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let r_e_prev = self.r_e;

        let drive_e = self.j_ee * self.r_e - self.j_ei * self.r_i + input;
        let drive_i = self.j_ie * self.r_e - self.j_ii * self.r_i;

        let dr_e = (-self.r_e + self.phi(drive_e)) / self.tau_e;
        let dr_i = (-self.r_i + self.phi(drive_i)) / self.tau_i;

        self.r_e += self.dt * dr_e;
        self.r_i += self.dt * dr_i;

        // Rates non-negative
        if self.r_e < 0.0 {
            self.r_e = 0.0;
        }
        if self.r_i < 0.0 {
            self.r_i = 0.0;
        }

        // Safety bounds
        if self.r_e > 200.0 {
            self.r_e = 200.0;
        }
        if self.r_i > 200.0 {
            self.r_i = 200.0;
        }
        if !self.r_e.is_finite() {
            self.r_e = 0.1;
        }
        if !self.r_i.is_finite() {
            self.r_i = 0.1;
        }

        // "Spike" when E rate crosses threshold
        if self.r_e >= self.r_threshold && r_e_prev < self.r_threshold {
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
    fn brunel_fires_with_input() {
        let mut n = BrunelNetwork::new();
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += n.step(5.0);
        }
        assert!(
            spikes > 0,
            "Brunel must produce bursts with input, got {spikes}"
        );
    }

    #[test]
    fn brunel_silent_without_input() {
        let mut n = BrunelNetwork::new();
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(
            spikes, 0,
            "Brunel must be quiescent without input, got {spikes}"
        );
    }

    #[test]
    fn brunel_ei_balance() {
        // Strong inhibition keeps E rate bounded
        let mut n = BrunelNetwork::new();
        for _ in 0..50_000 {
            n.step(3.0);
        }
        assert!(
            n.r_e < 50.0,
            "E/I balance should keep r_e bounded, r_e={}",
            n.r_e
        );
        assert!(n.r_i >= 0.0, "r_i must be non-negative");
    }

    #[test]
    fn brunel_inhibition_suppresses() {
        // Increasing j_ei should reduce E rate
        let mut weak_inh = BrunelNetwork::new();
        weak_inh.j_ei = 0.3;
        let mut strong_inh = BrunelNetwork::new();
        strong_inh.j_ei = 2.0;

        for _ in 0..20_000 {
            weak_inh.step(5.0);
            strong_inh.step(5.0);
        }
        assert!(
            weak_inh.r_e >= strong_inh.r_e,
            "Stronger inhibition → lower E rate: weak={:.2} vs strong={:.2}",
            weak_inh.r_e,
            strong_inh.r_e
        );
    }

    #[test]
    fn brunel_rates_non_negative() {
        let mut n = BrunelNetwork::new();
        for _ in 0..50_000 {
            n.step(-10.0);
        }
        assert!(n.r_e >= 0.0);
        assert!(n.r_i >= 0.0);
    }

    #[test]
    fn brunel_negative_input_no_crash() {
        let mut n = BrunelNetwork::new();
        for _ in 0..50_000 {
            n.step(-100.0);
        }
        assert!(n.r_e.is_finite());
        assert!(n.r_i.is_finite());
    }

    #[test]
    fn brunel_nan_input_stays_finite() {
        let mut n = BrunelNetwork::new();
        n.step(f64::NAN);
        assert!(n.r_e.is_finite());
        assert!(n.r_i.is_finite());
    }

    #[test]
    fn brunel_extreme_input_bounded() {
        let mut n = BrunelNetwork::new();
        for _ in 0..10_000 {
            n.step(1e6);
        }
        assert!(n.r_e.is_finite() && n.r_e <= 200.0);
    }

    #[test]
    fn brunel_reset_clears_state() {
        let mut n = BrunelNetwork::new();
        for _ in 0..10_000 {
            n.step(5.0);
        }
        n.reset();
        assert_eq!(n.r_e, 0.1);
        assert_eq!(n.r_i, 0.1);
    }

    #[test]
    fn brunel_performance_100k_steps() {
        let start = std::time::Instant::now();
        let mut n = BrunelNetwork::new();
        for _ in 0..100_000 {
            std::hint::black_box(n.step(3.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "100k steps must complete in <50ms"
        );
    }
}
