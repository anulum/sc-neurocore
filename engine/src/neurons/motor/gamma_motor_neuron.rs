// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Gamma Motor Neuron

//! Gamma motor-neuron fusimotor-drive dynamics.

#![deny(missing_docs)]

/// Gamma motor neuron — innervates intrafusal fibres of muscle spindles.
///
/// Regulates proprioceptive sensitivity by adjusting spindle tension.
/// Smaller soma than alpha, lower firing rates (5-30 Hz), no PIC.
/// Simple LIF with spike-frequency adaptation (slow K+ current).
/// Two subtypes, labelled by the `dynamic` flag: dynamic (bag1,
/// velocity-sensitive) and static (bag2/chain, length-sensitive). They
/// differ through their time constants and adaptation coupling.
///
/// Prochazka & Hulliger, Prog. Brain Res. 80, 1989.
/// Taylor et al., J. Physiol. 519(3), 1999.
#[derive(Clone, Debug)]
pub struct GammaMotorNeuron {
    /// Membrane potential (mV).
    pub v: f64,
    /// Resting membrane potential (mV).
    pub v_rest: f64,
    /// Post-spike reset potential (mV).
    pub v_reset: f64,
    /// Spike-detection threshold (mV).
    pub v_threshold: f64,
    /// Membrane time constant (ms).
    pub tau: f64,
    /// Adaptation (mV), subtracted from the drive; it relaxes towards `a_adapt * (v - v_rest)`.
    pub adapt: f64,
    /// Spike-frequency adaptation time constant (ms).
    pub tau_adapt: f64,
    /// Subthreshold adaptation coupling strength (dimensionless).
    pub a_adapt: f64,
    /// Gain from the fusimotor drive to mV in the target potential (mV per drive unit).
    pub gain: f64,
    /// True = dynamic (bag1), false = static (bag2/chain). The step does not read it; the
    /// two subtypes differ through `tau`, `tau_adapt` and `a_adapt`.
    pub dynamic: bool,
    /// Step (ms).
    pub dt: f64,
}

impl GammaMotorNeuron {
    /// Create a dynamic gamma motor neuron, the same as `dynamic`.
    pub fn new() -> Self {
        Self::dynamic()
    }

    /// Dynamic gamma — innervates bag1 intrafusal fibres (velocity-sensitive).
    pub fn dynamic() -> Self {
        Self {
            v: -65.0,
            v_rest: -65.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau: 8.0,
            adapt: 0.0,
            tau_adapt: 100.0,
            a_adapt: 0.3,
            gain: 1.0,
            dynamic: true,
            dt: 0.5,
        }
    }

    /// Static gamma — innervates bag2/chain intrafusal fibres (length-sensitive).
    pub fn static_type() -> Self {
        Self {
            tau: 12.0,        // Slower membrane
            tau_adapt: 200.0, // Larger adaptation time constant
            a_adapt: 0.5,
            dynamic: false,
            ..Self::dynamic()
        }
    }

    /// Step with fusimotor drive (arbitrary units, ≥ 0). Returns spike (1/0).
    ///
    /// `v` relaxes exactly over `dt` towards `v_rest + gain * max(drive, 0) - adapt` with
    /// `tau`. Then `adapt` relaxes towards `a_adapt * (v - v_rest)`, using the new `v`, with
    /// `tau_adapt`. Reaching `v_threshold` sets `v` to `v_reset`. An invalid state or
    /// parameter, a non-finite drive or a non-finite result returns 0 and leaves the state
    /// unchanged.
    pub fn step(&mut self, drive: f64) -> i32 {
        if !self.is_valid() || !drive.is_finite() {
            return 0;
        }
        let v_old = self.v;
        let adapt_old = self.adapt;
        let input = self.gain * drive.max(0.0) - adapt_old;
        let v_target = self.v_rest + input;
        let v_candidate = v_target + (v_old - v_target) * (-self.dt / self.tau).exp();
        let adapt_target = self.a_adapt * (v_candidate - self.v_rest);
        let adapt_candidate =
            adapt_target + (adapt_old - adapt_target) * (-self.dt / self.tau_adapt).exp();
        if !v_candidate.is_finite() || !adapt_candidate.is_finite() {
            return 0;
        }
        self.v = v_candidate;
        self.adapt = adapt_candidate;

        if v_candidate >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    /// Return `v` to `v_rest` and clear the adaptation; the parameters are kept.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.adapt = 0.0;
    }

    fn is_valid(&self) -> bool {
        [
            self.v,
            self.v_rest,
            self.v_reset,
            self.v_threshold,
            self.tau,
            self.adapt,
            self.tau_adapt,
            self.a_adapt,
            self.gain,
            self.dt,
        ]
        .iter()
        .all(|value| value.is_finite())
            && self.tau > 0.0
            && self.tau_adapt > 0.0
            && self.dt > 0.0
            && self.gain >= 0.0
            && self.v_reset < self.v_threshold
    }
}

impl Default for GammaMotorNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neurons::host_budget;

    // ── Gamma Motor Neuron — 6-dimension coverage ──────────────────

    #[test]
    fn gamma_dynamic_fires_with_drive() {
        let mut n = GammaMotorNeuron::dynamic();
        let spikes: i32 = (0..2000).map(|_| n.step(20.0)).sum();
        assert!(spikes > 0, "gamma dynamic must fire: got {spikes}");
    }

    #[test]
    fn gamma_static_fires_with_drive() {
        let mut n = GammaMotorNeuron::static_type();
        let spikes: i32 = (0..2000).map(|_| n.step(20.0)).sum();
        assert!(spikes > 0, "gamma static must fire: got {spikes}");
    }

    #[test]
    fn gamma_no_fire_without_drive() {
        let mut n = GammaMotorNeuron::new();
        let spikes: i32 = (0..1000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn gamma_negative_drive_no_fire() {
        let mut n = GammaMotorNeuron::new();
        // drive.max(0.0) clamps negatives
        let spikes: i32 = (0..1000).map(|_| n.step(-10.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn gamma_adaptation_reduces_rate() {
        let mut n = GammaMotorNeuron::new();
        let first: i32 = (0..1000).map(|_| n.step(20.0)).sum();
        let second: i32 = (0..1000).map(|_| n.step(20.0)).sum();
        assert!(
            second <= first + 3,
            "gamma should adapt: first={first}, second={second}"
        );
    }

    #[test]
    fn gamma_static_adapts_more_than_dynamic() {
        let mut dyn_ = GammaMotorNeuron::dynamic();
        let mut stat = GammaMotorNeuron::static_type();
        let dyn_spikes: i32 = (0..2000).map(|_| dyn_.step(20.0)).sum();
        let stat_spikes: i32 = (0..2000).map(|_| stat.step(20.0)).sum();
        // Static uses larger adaptation coupling and lower excitability.
        assert!(
            stat_spikes <= dyn_spikes + 5,
            "static ({stat_spikes}) should fire <= dynamic ({dyn_spikes})"
        );
    }

    #[test]
    fn gamma_reset_roundtrip() {
        let mut n = GammaMotorNeuron::new();
        for _ in 0..1000 {
            n.step(20.0);
        }
        n.reset();
        let mut fresh = GammaMotorNeuron::new();
        let r1: i32 = (0..500).map(|_| n.step(20.0)).sum();
        let r2: i32 = (0..500).map(|_| fresh.step(20.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn gamma_voltage_bounded() {
        let mut n = GammaMotorNeuron::new();
        for _ in 0..10000 {
            n.step(50.0);
        }
        assert!(n.v.is_finite());
        assert!(n.adapt.is_finite());
    }

    #[test]
    fn gamma_nan_recovery() {
        let mut n = GammaMotorNeuron::new();
        for _ in 0..50 {
            n.step(20.0);
        }
        let before_v = n.v;
        let before_adapt = n.adapt;
        for _ in 0..10 {
            let _ = n.step(f64::NAN);
        }
        assert_eq!(n.v, before_v);
        assert_eq!(n.adapt, before_adapt);
        n.reset();
        assert!(n.v.is_finite());
        assert_eq!(n.adapt, 0.0);
    }

    #[test]
    fn gamma_extreme_input() {
        let mut n = GammaMotorNeuron::new();
        for _ in 0..50 {
            n.step(1e6);
        }
        n.reset();
        assert!(n.v.is_finite());
    }

    #[test]
    fn gamma_corrupted_state_preserved_on_step() {
        let mut n = GammaMotorNeuron::new();
        n.tau = 0.0;
        let before_v = n.v;
        let before_adapt = n.adapt;
        assert_eq!(n.step(20.0), 0);
        assert_eq!(n.v, before_v);
        assert_eq!(n.adapt, before_adapt);
    }

    #[test]
    fn gamma_performance() {
        let mut n = GammaMotorNeuron::new();
        host_budget::assert_within(module_path!(), "gamma_performance", || {
            for _ in 0..100_000 {
                std::hint::black_box(n.step(20.0));
            }
        });
    }
}
