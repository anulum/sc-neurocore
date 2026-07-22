// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Motor Unit

//! Motor-unit neural drive and muscle-force dynamics.

/// Motor unit — functional unit of motor control: alpha motor neuron + muscle fibre.
///
/// Each spike from the embedded LIF motor neuron triggers a muscle twitch.
/// Force output is the summation of overlapping twitches (rate coding).
/// Higher firing rates → more twitch overlap → higher force (tetanus).
///
/// Muscle twitch modelled as a critically-damped second-order system:
/// f(t) = A * (t/τ) * exp(1 - t/τ), giving a smooth rise-then-decay.
///
/// Fuglevand et al., J. Neurophysiol. 70(6), 1993.
/// Heckman & Enoka, Compr. Physiol. 2(4), 2012.
#[derive(Clone, Debug)]
pub struct MotorUnit {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64, // Membrane time constant (ms)
    pub adapt: f64,
    pub tau_adapt: f64,
    pub a_adapt: f64,
    pub gain: f64,
    // Muscle fibre
    pub force: f64,       // Current force output (normalised)
    pub twitch_amp: f64,  // Peak twitch amplitude
    pub tau_twitch: f64,  // Twitch contraction time (ms)
    pub force_decay: f64, // Force decay per step
    pub dt: f64,
}

impl MotorUnit {
    pub fn new() -> Self {
        Self::slow()
    }

    /// Slow motor unit (type S): small, fatigue-resistant, low force.
    pub fn slow() -> Self {
        Self {
            v: -65.0,
            v_rest: -65.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau_m: 10.0,
            adapt: 0.0,
            tau_adapt: 100.0,
            a_adapt: 0.2,
            gain: 1.0,
            force: 0.0,
            twitch_amp: 0.05,
            tau_twitch: 90.0,
            force_decay: 0.0,
            dt: 0.5,
        }
    }

    /// Fast motor unit (type FF): large, fatigable, high force.
    pub fn fast() -> Self {
        Self {
            tau_m: 6.0,
            tau_adapt: 50.0,
            a_adapt: 0.1,
            twitch_amp: 0.3,
            tau_twitch: 30.0,
            ..Self::slow()
        }
    }

    fn voltage_valid(value: f64) -> bool {
        value.is_finite() && (-150.0..=100.0).contains(&value)
    }

    fn force_valid(value: f64) -> bool {
        value.is_finite() && (0.0..=1.0).contains(&value)
    }

    fn exact_relax(previous: f64, steady: f64, tau: f64, dt: f64) -> Option<f64> {
        if !previous.is_finite()
            || !steady.is_finite()
            || !tau.is_finite()
            || !dt.is_finite()
            || tau <= 0.0
            || dt <= 0.0
        {
            return None;
        }
        Some(steady + (previous - steady) * (-dt / tau).exp())
    }

    fn valid_state(&self) -> bool {
        Self::voltage_valid(self.v)
            && Self::voltage_valid(self.v_rest)
            && Self::voltage_valid(self.v_reset)
            && Self::voltage_valid(self.v_threshold)
            && Self::force_valid(self.force)
            && self.tau_m.is_finite()
            && self.adapt.is_finite()
            && self.tau_adapt.is_finite()
            && self.a_adapt.is_finite()
            && self.gain.is_finite()
            && self.twitch_amp.is_finite()
            && self.tau_twitch.is_finite()
            && self.force_decay.is_finite()
            && self.dt.is_finite()
            && self.tau_m > 0.0
            && self.tau_adapt > 0.0
            && self.tau_twitch > 0.0
            && self.dt > 0.0
            && self.gain >= 0.0
            && self.twitch_amp >= 0.0
            && self.v_reset < self.v_threshold
    }

    /// Step with descending drive (≥ 0). Returns spike (1/0). Force accessible via `.force`.
    pub fn step(&mut self, drive: f64) -> i32 {
        if !drive.is_finite() || !self.valid_state() {
            return 0;
        }

        let mut force = self.force * (-self.dt / self.tau_twitch).exp();
        let input = self.gain * drive.max(0.0) - self.adapt;
        let v_target = self.v_rest + input;
        let Some(mut v_candidate) = Self::exact_relax(self.v, v_target, self.tau_m, self.dt) else {
            return 0;
        };
        if !Self::voltage_valid(v_candidate) {
            return 0;
        }

        let adapt_target = self.a_adapt * (v_candidate - self.v_rest);
        let Some(adapt_candidate) =
            Self::exact_relax(self.adapt, adapt_target, self.tau_adapt, self.dt)
        else {
            return 0;
        };
        if !adapt_candidate.is_finite() {
            return 0;
        }

        let mut spike = 0;
        if v_candidate >= self.v_threshold {
            v_candidate = self.v_reset;
            force = (force + self.twitch_amp).min(1.0);
            spike = 1;
        }
        if !Self::voltage_valid(v_candidate) || !Self::force_valid(force) {
            return 0;
        }

        self.v = v_candidate;
        self.adapt = adapt_candidate;
        self.force = force;
        spike
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.adapt = 0.0;
        self.force = 0.0;
    }
}

impl Default for MotorUnit {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Motor Unit — 6-dimension coverage ──────────────────────────

    #[test]
    fn motor_unit_fires_with_drive() {
        let mut mu = MotorUnit::new();
        let spikes: i32 = (0..2000).map(|_| mu.step(20.0)).sum();
        assert!(spikes > 0, "motor unit must fire: got {spikes}");
    }

    #[test]
    fn motor_unit_no_fire_without_drive() {
        let mut mu = MotorUnit::new();
        let spikes: i32 = (0..1000).map(|_| mu.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn motor_unit_negative_drive_no_fire() {
        let mut mu = MotorUnit::new();
        let spikes: i32 = (0..1000).map(|_| mu.step(-10.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn motor_unit_force_increases_with_spikes() {
        let mut mu = MotorUnit::new();
        assert_eq!(mu.force, 0.0);
        for _ in 0..2000 {
            mu.step(20.0);
        }
        assert!(
            mu.force > 0.0,
            "force should increase during spiking: f={}",
            mu.force
        );
    }

    #[test]
    fn motor_unit_force_decays_without_input() {
        let mut mu = MotorUnit::new();
        // Build up force
        for _ in 0..1000 {
            mu.step(20.0);
        }
        let peak = mu.force;
        assert!(peak > 0.0);
        // No input → force decays
        for _ in 0..5000 {
            mu.step(0.0);
        }
        assert!(
            mu.force < peak,
            "force should decay: peak={peak}, now={}",
            mu.force
        );
    }

    #[test]
    fn motor_unit_fast_produces_more_force() {
        let mut slow = MotorUnit::slow();
        let mut fast = MotorUnit::fast();
        for _ in 0..2000 {
            slow.step(20.0);
            fast.step(20.0);
        }
        assert!(
            fast.force >= slow.force,
            "fast MU ({}) should produce >= force than slow ({})",
            fast.force,
            slow.force
        );
    }

    #[test]
    fn motor_unit_force_capped_at_one() {
        let mut mu = MotorUnit::fast();
        for _ in 0..10000 {
            mu.step(50.0);
        }
        assert!(mu.force <= 1.0, "force must not exceed 1.0: f={}", mu.force);
    }

    #[test]
    fn motor_unit_reset_roundtrip() {
        let mut mu = MotorUnit::new();
        for _ in 0..1000 {
            mu.step(20.0);
        }
        mu.reset();
        assert_eq!(mu.force, 0.0);
        assert_eq!(mu.adapt, 0.0);
        let mut fresh = MotorUnit::new();
        let r1: i32 = (0..500).map(|_| mu.step(20.0)).sum();
        let r2: i32 = (0..500).map(|_| fresh.step(20.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn motor_unit_voltage_bounded() {
        let mut mu = MotorUnit::new();
        for _ in 0..10000 {
            mu.step(50.0);
        }
        assert!(mu.v.is_finite());
        assert!(mu.force.is_finite());
    }

    #[test]
    fn motor_unit_nan_recovery() {
        let mut mu = MotorUnit::new();
        for _ in 0..50 {
            mu.step(20.0);
        }
        for _ in 0..10 {
            let _ = mu.step(f64::NAN);
        }
        mu.reset();
        assert!(mu.v.is_finite());
        assert_eq!(mu.force, 0.0);
    }

    #[test]
    fn motor_unit_performance() {
        let mut mu = MotorUnit::new();
        let start = std::time::Instant::now();
        for _ in 0..100_000 {
            mu.step(20.0);
        }
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 100, "100k steps took {:?}", elapsed);
    }
}
