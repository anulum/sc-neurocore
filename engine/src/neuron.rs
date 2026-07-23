// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Neuron Models

//! # Neuron Models
//!
//! Fixed-point LIF and Izhikevich neuron models for the v3 engine.

mod adex;
mod bitstream_averager;
mod dendritic_neuron;
mod homeostatic_lif;
mod izhikevich;

pub use adex::AdExNeuron;
pub use bitstream_averager::BitstreamAverager;
pub use dendritic_neuron::DendriticNeuron;
pub use homeostatic_lif::HomeostaticLif;
pub use izhikevich::Izhikevich;

/// Mask and sign-interpret an integer to `width` bits (branchless).
///
/// `width` must be in 1..=32. Values outside this range trigger a
/// debug assertion failure (release builds silently produce garbage).
#[inline]
pub fn mask(value: i32, width: u32) -> i16 {
    assert!(
        width > 0 && width <= 32,
        "mask width must be 1..=32, got {width}"
    );
    let m = (1_i64 << width) - 1;
    let v = (value as i64) & m;
    let shift = 64 - width;
    ((v << shift) >> shift) as i16
}

/// Fixed-point leaky-integrate-and-fire neuron state and parameters.
#[derive(Clone, Debug)]
pub struct FixedPointLif {
    /// Membrane potential.
    pub v: i16,
    /// Refractory counter in simulation steps.
    pub refractory_counter: i32,
    /// Arithmetic data width.
    pub data_width: u32,
    /// Fraction bits for fixed-point scaling.
    pub fraction: u32,
    /// Resting potential.
    pub v_rest: i16,
    /// Reset potential after spike.
    pub v_reset: i16,
    /// Spike threshold.
    pub v_threshold: i16,
    /// Refractory period length in steps.
    pub refractory_period: i32,
}

impl FixedPointLif {
    /// Construct a fixed-point LIF neuron.
    pub fn new(
        data_width: u32,
        fraction: u32,
        v_rest: i16,
        v_reset: i16,
        v_threshold: i16,
        refractory_period: i32,
    ) -> Self {
        Self {
            v: v_rest,
            refractory_counter: 0,
            data_width,
            fraction,
            v_rest,
            v_reset,
            v_threshold,
            refractory_period,
        }
    }

    /// Advance one simulation step.
    ///
    /// Returns `(spike, membrane_voltage)`.
    #[allow(non_snake_case)]
    pub fn step(&mut self, leak_k: i16, gain_k: i16, i_t: i16, noise_in: i16) -> (i32, i16) {
        let w = self.data_width;

        // Refractory: check previous step's counter before any fire logic.
        if self.refractory_counter > 0 {
            self.refractory_counter -= 1;
            self.v = self.v_rest;
            return (0, mask(self.v_rest as i32, w));
        }

        let diff = mask((self.v_rest as i32) - (self.v as i32), 2 * w) as i32;
        let dv_leak = mask((diff * (leak_k as i32)) >> self.fraction, self.data_width);
        let dv_in = mask(
            ((i_t as i32) * (gain_k as i32)) >> self.fraction,
            self.data_width,
        );

        let v_next = mask(
            (self.v as i32) + (dv_leak as i32) + (dv_in as i32) + (noise_in as i32),
            self.data_width,
        );

        if v_next >= self.v_threshold {
            self.v = self.v_reset;
            self.refractory_counter = self.refractory_period;
            (1, mask(self.v_reset as i32, w))
        } else {
            self.v = v_next;
            (0, mask(v_next as i32, w))
        }
    }

    /// Reset internal state to resting potential.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.refractory_counter = 0;
    }
}

/// Exponential IF (no adaptation). Fourcaud-Trocmé et al. 2003.
#[derive(Clone, Debug)]
pub struct ExpIfNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub v_rh: f64,
    pub delta_t: f64,
    pub tau: f64,
    pub dt: f64,
    pub refractory_period: f64,
    pub refractory_remaining: f64,
    /// Precomputed 1.0 / delta_t.
    pub inv_delta_t: f64,
    /// Precomputed dt / tau.
    pub dt_div_tau: f64,
}

impl Default for ExpIfNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpIfNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            v_rest: -65.0,
            v_reset: -68.0,
            v_threshold: 30.0,
            v_rh: -59.9,
            delta_t: 3.48,
            tau: 10.0,
            dt: 0.02,
            refractory_period: 0.0,
            refractory_remaining: 0.0,
            inv_delta_t: 1.0 / 3.48,
            dt_div_tau: 0.02 / 10.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !self.v.is_finite()
            || !current.is_finite()
            || !self.v_rest.is_finite()
            || !self.v_reset.is_finite()
            || !self.v_threshold.is_finite()
            || !self.v_rh.is_finite()
            || !self.delta_t.is_finite()
            || !self.tau.is_finite()
            || !self.dt.is_finite()
            || !self.refractory_period.is_finite()
            || !self.refractory_remaining.is_finite()
            || self.delta_t <= 0.0
            || self.tau <= 0.0
            || self.dt <= 0.0
            || self.refractory_period < 0.0
            || self.refractory_remaining < 0.0
            || self.refractory_remaining > self.refractory_period
            || self.v_threshold <= self.v_rh
            || self.v >= self.v_threshold
            || self.v_rest >= self.v_threshold
            || self.v_reset >= self.v_threshold
        {
            return 0;
        }

        if self.refractory_remaining > 0.0 {
            self.refractory_remaining = (self.refractory_remaining - self.dt).max(0.0);
            self.v = self.v_reset;
            return 0;
        }

        let inv_delta_t = 1.0 / self.delta_t;
        let k1 = self.rhs(self.v, current, inv_delta_t);
        let k2 = self.rhs(self.v + 0.5 * self.dt * k1, current, inv_delta_t);
        let k3 = self.rhs(self.v + 0.5 * self.dt * k2, current, inv_delta_t);
        let k4 = self.rhs(self.v + self.dt * k3, current, inv_delta_t);
        let next_v = self.v + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        if !k1.is_finite()
            || !k2.is_finite()
            || !k3.is_finite()
            || !k4.is_finite()
            || !next_v.is_finite()
        {
            return 0;
        }

        self.inv_delta_t = inv_delta_t;
        self.dt_div_tau = self.dt / self.tau;
        if next_v >= self.v_threshold {
            self.v = self.v_reset;
            self.refractory_remaining = self.refractory_period;
            1
        } else {
            self.v = next_v;
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.refractory_remaining = 0.0;
    }

    fn rhs(&self, v: f64, current: f64, inv_delta_t: f64) -> f64 {
        if !v.is_finite() {
            return f64::NAN;
        }
        let bounded_v = v.min(self.v_threshold);
        let exp_arg = (bounded_v - self.v_rh) * inv_delta_t;
        let exp_term = self.delta_t * exp_arg.exp();
        (-(bounded_v - self.v_rest) + exp_term + current) / self.tau
    }
}

/// Lapicque 1907 — classical RC integrate-and-fire.
#[derive(Clone, Debug)]
pub struct LapicqueNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau: f64,
    pub resistance: f64,
    pub dt: f64,
}

impl LapicqueNeuron {
    pub fn new(tau: f64, resistance: f64, threshold: f64, dt: f64) -> Self {
        Self {
            v: 0.0,
            v_rest: 0.0,
            v_reset: 0.0,
            v_threshold: threshold,
            tau,
            resistance,
            dt,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !self.v.is_finite()
            || !self.v_rest.is_finite()
            || !self.v_reset.is_finite()
            || !self.v_threshold.is_finite()
            || self.v_threshold <= self.v_rest
            || self.v_threshold <= self.v_reset
            || self.v >= self.v_threshold
            || !self.tau.is_finite()
            || self.tau <= 0.0
            || !self.resistance.is_finite()
            || self.resistance <= 0.0
            || !self.dt.is_finite()
            || self.dt <= 0.0
            || !current.is_finite()
        {
            return 0;
        }

        let v_inf = self.v_rest + self.resistance * current;
        let decay = (-self.dt / self.tau).exp();
        let next_v = v_inf + (self.v - v_inf) * decay;
        if !v_inf.is_finite() || !decay.is_finite() || !next_v.is_finite() {
            return 0;
        }
        self.v = next_v;

        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_exp_if_optimisation_parity() {
        let mut n = ExpIfNeuron::new();
        n.v = -60.0;
        let current = 10.0;

        let rhs = |v: f64| {
            let bounded_v = v.min(n.v_threshold);
            let exp_arg = (bounded_v - n.v_rh) / n.delta_t;
            (-(bounded_v - n.v_rest) + n.delta_t * exp_arg.exp() + current) / n.tau
        };
        let k1 = rhs(n.v);
        let k2 = rhs(n.v + 0.5 * n.dt * k1);
        let k3 = rhs(n.v + 0.5 * n.dt * k2);
        let k4 = rhs(n.v + n.dt * k3);
        let expected_dv = (n.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

        n.step(current);
        let got_dv = n.v - (-60.0); // Simple check since we only did one step

        // Use a small epsilon for float parity.
        assert!(
            (got_dv - expected_dv).abs() < 1e-12,
            "Logic mismatch in ExpIfNeuron: got {}, expected {}",
            got_dv,
            expected_dv
        );
    }

    use super::{mask, ExpIfNeuron, FixedPointLif, LapicqueNeuron};

    #[test]
    fn mask_branchless_matches_original() {
        for &width in &[16_u32, 32] {
            for value in [
                -32768_i32,
                -1,
                0,
                1,
                32767,
                65535,
                -65536,
                i16::MAX as i32,
                i16::MIN as i32,
            ] {
                let result = mask(value, width);

                let m = (1_i64 << width) - 1;
                let mut v = (value as i64) & m;
                if v >= (1_i64 << (width - 1)) {
                    v -= 1_i64 << width;
                }
                let expected = if width >= 32 {
                    v as i32 as i16
                } else {
                    v as i16
                };

                assert_eq!(
                    result, expected,
                    "mask({value}, {width}): got {result}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn lif_fires_with_refractory_period() {
        // Q8.8: threshold=1.0 → 256, matching Python default
        let mut n = FixedPointLif::new(16, 8, 0, 0, 256, 2);
        let mut spikes = Vec::new();
        for _ in 0..30 {
            let (s, _) = n.step(1, 256, 50, 0);
            spikes.push(s);
        }
        let total: i32 = spikes.iter().sum();
        assert!(total > 0, "neuron must fire with refractory_period=2");
        // Refractory gap: after a spike, next 2 steps must be silent.
        for (i, &s) in spikes.iter().enumerate() {
            if s == 1 && i + 2 < spikes.len() {
                assert_eq!(spikes[i + 1], 0, "step {} should be refractory", i + 1);
                assert_eq!(spikes[i + 2], 0, "step {} should be refractory", i + 2);
            }
        }
    }

    #[test]
    fn lif_fires_without_refractory() {
        let mut n = FixedPointLif::new(16, 8, 0, 0, 256, 0);
        let mut total = 0;
        for _ in 0..20 {
            let (s, _) = n.step(1, 256, 50, 0);
            total += s;
        }
        assert!(total > 0, "neuron must fire with refractory_period=0");
    }

    // ── ExpIF tests ───────────────────────────────────────────────

    #[test]
    fn expif_fires() {
        let mut n = ExpIfNeuron::new();
        let mut total = 0;
        for _ in 0..2000 {
            total += n.step(500.0);
        }
        assert!(total > 0, "ExpIF must fire");
    }

    #[test]
    fn expif_no_fire_without_input() {
        let mut n = ExpIfNeuron::new();
        let total: i32 = (0..500).map(|_| n.step(0.0)).sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn expif_matches_rk4_reference() {
        let mut n = ExpIfNeuron::new();
        n.v = -60.0;
        n.dt = 0.25;
        n.tau = 20.0;
        let rhs = |v: f64, s: &ExpIfNeuron, current: f64| {
            let bounded_v = v.min(s.v_threshold);
            let exp_arg = (bounded_v - s.v_rh) / s.delta_t;
            (-(bounded_v - s.v_rest) + s.delta_t * exp_arg.exp() + current) / s.tau
        };
        let current = 12.0;
        let k1 = rhs(n.v, &n, current);
        let k2 = rhs(n.v + 0.5 * n.dt * k1, &n, current);
        let k3 = rhs(n.v + 0.5 * n.dt * k2, &n, current);
        let k4 = rhs(n.v + n.dt * k3, &n, current);
        let expected = n.v + (n.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

        assert_eq!(n.step(current), 0);
        assert!((n.v - expected).abs() < 1e-12);
    }

    // ── Lapicque tests ────────────────────────────────────────────

    #[test]
    fn lapicque_fires() {
        let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
        let mut total = 0;
        for _ in 0..200 {
            total += n.step(5.0);
        }
        assert!(total > 0, "Lapicque must fire with sustained input");
    }

    #[test]
    fn lapicque_reset() {
        let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
        for _ in 0..50 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.v).abs() < 1e-12);
    }

    #[test]
    fn lapicque_exact_flow_matches_closed_form() {
        let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 5.0);
        n.v = 0.25;
        let current = 0.5;
        let v0 = n.v;
        let v_inf = n.v_rest + n.resistance * current;
        let euler = v0 + (-(v0 - n.v_rest) + n.resistance * current) / n.tau * n.dt;
        let expected = v_inf + (v0 - v_inf) * (-n.dt / n.tau).exp();

        assert_eq!(n.step(current), 0);
        assert!((n.v - expected).abs() < 1e-15);
        assert!((n.v - euler).abs() > 1e-4);
    }

    // ── ExpIF coverage tests ───────────────────────────────────────

    #[test]
    fn expif_negative_current_no_fire() {
        let mut n = ExpIfNeuron::new();
        let total: i32 = (0..500).map(|_| n.step(-100.0)).sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn expif_reset_roundtrip() {
        let mut n = ExpIfNeuron::new();
        for _ in 0..200 {
            n.step(500.0);
        }
        n.reset();
        assert_eq!(n.v, n.v_rest);
        let mut fresh = ExpIfNeuron::new();
        let r1: i32 = (0..100).map(|_| n.step(500.0)).sum();
        let r2: i32 = (0..100).map(|_| fresh.step(500.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn expif_voltage_bounded() {
        let mut n = ExpIfNeuron::new();
        for _ in 0..5000 {
            n.step(1000.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn expif_enrolled_event_counts() {
        for (current, expected) in [(0.0, 0), (5.0, 0), (20.0, 2)] {
            let mut neuron = ExpIfNeuron::new();
            let spikes: i32 = (0..1000).map(|_| neuron.step(current)).sum();
            assert_eq!(spikes, expected, "current={current}");
        }
    }

    #[test]
    fn expif_refractory_hold_and_fail_closed_state() {
        let mut neuron = ExpIfNeuron::new();
        neuron.refractory_period = 1.7;
        while neuron.step(50.0) == 0 {}
        assert_eq!(neuron.refractory_remaining, 1.7);
        for _ in 0..10 {
            assert_eq!(neuron.step(50.0), 0);
            assert_eq!(neuron.v, neuron.v_reset);
        }
        assert!((neuron.refractory_remaining - 1.5).abs() < 1.0e-12);

        let before = (neuron.v, neuron.refractory_remaining);
        neuron.refractory_remaining = 2.0;
        assert_eq!(neuron.step(0.0), 0);
        assert_eq!((neuron.v, neuron.refractory_remaining), (before.0, 2.0));
    }

    #[test]
    fn expif_performance_10k_steps() {
        let mut n = ExpIfNeuron::new();
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            n.step(500.0);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "10k steps took too long: {:?}",
            elapsed
        );
    }

    // ── Lapicque coverage tests ────────────────────────────────────

    #[test]
    fn lapicque_no_fire_without_input() {
        let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
        let total: i32 = (0..500).map(|_| n.step(0.0)).sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn lapicque_negative_current_no_fire() {
        let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
        let total: i32 = (0..500).map(|_| n.step(-5.0)).sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn lapicque_invalid_state_does_not_mutate() {
        let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
        n.v = 0.25;
        n.tau = 0.0;
        assert_eq!(n.step(1.0), 0);
        assert_eq!(n.v, 0.25);
    }

    #[test]
    fn lapicque_reset_roundtrip() {
        let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
        for _ in 0..100 {
            n.step(5.0);
        }
        n.reset();
        assert_eq!(n.v, n.v_rest);
        let mut fresh = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
        let r1: i32 = (0..100).map(|_| n.step(5.0)).sum();
        let r2: i32 = (0..100).map(|_| fresh.step(5.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn lapicque_voltage_bounded() {
        let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
        for _ in 0..5000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn lapicque_higher_resistance_fires_faster() {
        let mut lo = LapicqueNeuron::new(20.0, 0.5, 1.0, 1.0);
        let mut hi = LapicqueNeuron::new(20.0, 2.0, 1.0, 1.0);
        let lo_spikes: i32 = (0..200).map(|_| lo.step(1.0)).sum();
        let hi_spikes: i32 = (0..200).map(|_| hi.step(1.0)).sum();
        assert!(
            hi_spikes >= lo_spikes,
            "higher R ({hi_spikes}) should fire >= lower R ({lo_spikes})"
        );
    }

    #[test]
    fn lapicque_performance_10k_steps() {
        let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            n.step(5.0);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "10k steps took too long: {:?}",
            elapsed
        );
    }

    #[test]
    fn lapicque_pipeline_sustained_spiking() {
        let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
        let spikes: i32 = (0..10000).map(|_| n.step(5.0)).sum();
        assert!(
            spikes > 100,
            "sustained input should produce many spikes: got {spikes}"
        );
    }
}
