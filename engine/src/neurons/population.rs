// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Population and Mean-Field Neuron Models

//! Population-level and mean-field neuron models.
//!
//! Phase 3G: Montbrio-Pazo-Roxin, Brunel balanced network,
//! Tsodyks-Uziel-Markram, El Boustani network.
//! Added one by one with full 7-point checklist verification.

// ═══════════════════════════════════════════════════════════════════
// Montbrio-Pazo-Roxin (MPR) Mean-Field
// ═══════════════════════════════════════════════════════════════════

/// Montbrio-Pazo-Roxin 2015 — exact mean-field of QIF neuron population.
///
/// Reduces an infinite population of quadratic integrate-and-fire (QIF)
/// neurons to 2 ODEs for the population firing rate r and mean membrane
/// potential v. This is mathematically exact (not an approximation) for
/// Lorentzian-distributed heterogeneity.
///
/// dr/dt = (delta / (pi * tau^2)) + (2 * r * v / tau)
/// dv/dt = (v^2 + eta + I - (pi * tau * r)^2) / tau
///
/// where delta = heterogeneity width, eta = mean excitability,
/// tau = membrane time constant, I = external input.
///
/// Spike emitted when r exceeds a threshold (proxy for population burst).
///
/// Montbrio, Pazo & Roxin, Phys Rev X 5:021028, 2015.
#[derive(Clone, Debug)]
pub struct MontbrioMeanField {
    pub r: f64,         // Population firing rate (Hz)
    pub v: f64,         // Mean membrane potential
    pub delta: f64,     // Heterogeneity width (Lorentzian)
    pub eta: f64,       // Mean excitability
    pub tau: f64,       // Membrane time constant (ms)
    pub j: f64,         // Synaptic coupling strength
    pub dt: f64,
    pub r_threshold: f64,
    pub gain: f64,
}

impl Default for MontbrioMeanField {
    fn default() -> Self { Self::new() }
}

impl MontbrioMeanField {
    pub fn new() -> Self {
        Self {
            r: 0.01,
            v: -2.0,
            delta: 1.0,
            eta: -5.0,      // Below threshold for spontaneous activity
            tau: 1.0,
            j: 15.0,        // Excitatory coupling
            dt: 0.01,       // Small dt for stability
            r_threshold: 0.5,
            gain: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let r_prev = self.r;

        let pi = std::f64::consts::PI;
        let tau = self.tau;

        // MPR equations with synaptic coupling (j * r adds recurrence)
        let dr = (self.delta / (pi * tau * tau))
            + (2.0 * self.r * self.v / tau);
        let dv = (self.v * self.v + self.eta + input + self.j * tau * self.r
            - (pi * tau * self.r).powi(2)) / tau;

        self.r += self.dt * dr;
        self.v += self.dt * dv;

        // Rate must be non-negative
        if self.r < 0.0 { self.r = 0.0; }

        // Safety bounds
        if self.r > 100.0 { self.r = 100.0; }
        if self.v < -50.0 { self.v = -50.0; }
        if self.v > 50.0 { self.v = 50.0; }
        if !self.r.is_finite() { self.r = 0.01; }
        if !self.v.is_finite() { self.v = -2.0; }

        // "Spike" = population burst: r crosses threshold
        if self.r >= self.r_threshold && r_prev < self.r_threshold { 1 } else { 0 }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Brunel Balanced Network
// ═══════════════════════════════════════════════════════════════════

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
    fn default() -> Self { Self::new() }
}

impl BrunelNetwork {
    pub fn new() -> Self {
        Self {
            r_e: 0.1,
            r_i: 0.1,
            tau_e: 20.0,
            tau_i: 10.0,
            j_ee: 0.2,
            j_ei: 0.8,     // Strong I→E inhibition
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
        if self.r_e < 0.0 { self.r_e = 0.0; }
        if self.r_i < 0.0 { self.r_i = 0.0; }

        // Safety bounds
        if self.r_e > 200.0 { self.r_e = 200.0; }
        if self.r_i > 200.0 { self.r_i = 200.0; }
        if !self.r_e.is_finite() { self.r_e = 0.1; }
        if !self.r_i.is_finite() { self.r_i = 0.1; }

        // "Spike" when E rate crosses threshold
        if self.r_e >= self.r_threshold && r_e_prev < self.r_threshold { 1 } else { 0 }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpr_fires_with_input() {
        let mut n = MontbrioMeanField::new();
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += n.step(10.0);
        }
        assert!(spikes > 0, "MPR must produce bursts with strong input, got {spikes}");
    }

    #[test]
    fn mpr_silent_without_input() {
        // eta = -5 (below threshold), no input → quiescent
        let mut n = MontbrioMeanField::new();
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(spikes, 0, "MPR must be quiescent without input (eta<0), got {spikes}");
    }

    #[test]
    fn mpr_rate_increases_with_input() {
        let mut low = MontbrioMeanField::new();
        let mut high = MontbrioMeanField::new();
        for _ in 0..10_000 {
            low.step(3.0);
            high.step(15.0);
        }
        assert!(high.r > low.r,
            "Higher input → higher rate: high={:.3} vs low={:.3}", high.r, low.r);
    }

    #[test]
    fn mpr_two_ode_dynamics() {
        // Verify both r and v evolve from initial conditions
        let mut n = MontbrioMeanField::new();
        let r0 = n.r;
        let v0 = n.v;
        for _ in 0..1000 {
            n.step(5.0);
        }
        assert!(n.r != r0 || n.v != v0, "State must evolve from initial conditions");
    }

    #[test]
    fn mpr_rate_non_negative() {
        let mut n = MontbrioMeanField::new();
        for _ in 0..50_000 {
            n.step(-10.0);
        }
        assert!(n.r >= 0.0, "Rate must be non-negative, r={}", n.r);
    }

    #[test]
    fn mpr_negative_input_no_crash() {
        let mut n = MontbrioMeanField::new();
        for _ in 0..50_000 { n.step(-100.0); }
        assert!(n.r.is_finite());
        assert!(n.v.is_finite());
    }

    #[test]
    fn mpr_nan_input_stays_finite() {
        let mut n = MontbrioMeanField::new();
        n.step(f64::NAN);
        assert!(n.r.is_finite());
        assert!(n.v.is_finite());
    }

    #[test]
    fn mpr_extreme_input_bounded() {
        let mut n = MontbrioMeanField::new();
        for _ in 0..10_000 { n.step(1e6); }
        assert!(n.r.is_finite() && n.r <= 100.0);
        assert!(n.v.is_finite() && n.v <= 50.0);
    }

    #[test]
    fn mpr_reset_clears_state() {
        let mut n = MontbrioMeanField::new();
        for _ in 0..10_000 { n.step(10.0); }
        n.reset();
        assert_eq!(n.r, 0.01);
        assert_eq!(n.v, -2.0);
    }

    #[test]
    fn mpr_performance_100k_steps() {
        let start = std::time::Instant::now();
        let mut n = MontbrioMeanField::new();
        for _ in 0..100_000 { std::hint::black_box(n.step(5.0)); }
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 50, "100k steps must complete in <50ms");
    }

    // -- Brunel Balanced Network tests --

    #[test]
    fn brunel_fires_with_input() {
        let mut n = BrunelNetwork::new();
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += n.step(5.0);
        }
        assert!(spikes > 0, "Brunel must produce bursts with input, got {spikes}");
    }

    #[test]
    fn brunel_silent_without_input() {
        let mut n = BrunelNetwork::new();
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(spikes, 0, "Brunel must be quiescent without input, got {spikes}");
    }

    #[test]
    fn brunel_ei_balance() {
        // Strong inhibition keeps E rate bounded
        let mut n = BrunelNetwork::new();
        for _ in 0..50_000 {
            n.step(3.0);
        }
        assert!(n.r_e < 50.0, "E/I balance should keep r_e bounded, r_e={}", n.r_e);
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
        assert!(weak_inh.r_e >= strong_inh.r_e,
            "Stronger inhibition → lower E rate: weak={:.2} vs strong={:.2}",
            weak_inh.r_e, strong_inh.r_e);
    }

    #[test]
    fn brunel_rates_non_negative() {
        let mut n = BrunelNetwork::new();
        for _ in 0..50_000 { n.step(-10.0); }
        assert!(n.r_e >= 0.0);
        assert!(n.r_i >= 0.0);
    }

    #[test]
    fn brunel_negative_input_no_crash() {
        let mut n = BrunelNetwork::new();
        for _ in 0..50_000 { n.step(-100.0); }
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
        for _ in 0..10_000 { n.step(1e6); }
        assert!(n.r_e.is_finite() && n.r_e <= 200.0);
    }

    #[test]
    fn brunel_reset_clears_state() {
        let mut n = BrunelNetwork::new();
        for _ in 0..10_000 { n.step(5.0); }
        n.reset();
        assert_eq!(n.r_e, 0.1);
        assert_eq!(n.r_i, 0.1);
    }

    #[test]
    fn brunel_performance_100k_steps() {
        let start = std::time::Instant::now();
        let mut n = BrunelNetwork::new();
        for _ in 0..100_000 { std::hint::black_box(n.step(3.0)); }
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 50, "100k steps must complete in <50ms");
    }
}
