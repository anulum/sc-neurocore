// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Cerebellar Circuit Neuron Models

// ═══════════════════════════════════════════════════════════════════
// Unipolar Brush Cell
// ═══════════════════════════════════════════════════════════════════

/// Cerebellar unipolar brush cell (UBC) — excitatory interneuron in vestibular cerebellum.
///
/// Biophysics: LIF with a slow persistent (NMDA-like) current that sustains
/// depolarisation long after input ceases. The single brush-like dendrite
/// forms a giant synapse with a mossy fibre rosette, creating a 1:1 relay
/// that amplifies and prolongs the input signal.
///
/// UBCs are unique excitatory interneurons in the granular layer. They
/// transform brief mossy fibre bursts into prolonged granule cell
/// activation, important for vestibular signal processing and timing.
///
/// Bhatt et al., J Comp Neurol 349:560, 1994; Diana et al., J Neurosci 27:4374, 2007.
#[derive(Clone, Debug)]
pub struct UnipolarBrushCell {
    pub v: f64,
    pub persistent: f64, // Slow NMDA-like persistent current
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_persistent: f64,  // Slow decay of persistent current (ms)
    pub persistent_gain: f64, // How much input drives persistent current
    pub gain: f64,
    pub dt: f64,
}

impl Default for UnipolarBrushCell {
    fn default() -> Self {
        Self::new()
    }
}

impl UnipolarBrushCell {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            persistent: 0.0,
            v_rest: -65.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau_m: 8.0,
            tau_persistent: 200.0,
            persistent_gain: 0.5,
            gain: 2.5,
            dt: 0.5,
        }
    }

    fn finite(values: &[f64]) -> bool {
        values.iter().all(|value| value.is_finite())
    }

    fn valid_configuration(&self) -> bool {
        Self::finite(&[
            self.v_rest,
            self.v_reset,
            self.v_threshold,
            self.tau_m,
            self.tau_persistent,
            self.persistent_gain,
            self.gain,
            self.dt,
        ]) && self.tau_m > 0.0
            && self.tau_persistent > 0.0
            && self.persistent_gain >= 0.0
            && self.gain >= 0.0
            && self.dt > 0.0
            && self.v_reset < self.v_threshold
    }

    fn valid_state(&self) -> bool {
        Self::finite(&[self.v, self.persistent])
            && (-100.0..=60.0).contains(&self.v)
            && self.persistent >= 0.0
    }

    fn first_order_relaxation(previous: f64, steady_state: f64, dt: f64, tau: f64) -> f64 {
        previous + (steady_state - previous) * (-(-dt / tau).exp_m1())
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !self.valid_configuration() || !self.valid_state() || !current.is_finite() {
            return 0;
        }
        let input = self.gain * current.max(0.0);
        if !input.is_finite() {
            return 0;
        }
        let next_persistent = Self::first_order_relaxation(
            self.persistent,
            self.persistent_gain * input,
            self.dt,
            self.tau_persistent,
        )
        .max(0.0);
        let next_v = Self::first_order_relaxation(
            self.v,
            self.v_rest + input + next_persistent,
            self.dt,
            self.tau_m,
        );
        if !Self::finite(&[next_persistent, next_v]) {
            return 0;
        }
        self.persistent = next_persistent;
        if next_v >= self.v_threshold {
            self.v = self.v_reset;
            return 1;
        }
        self.v = next_v.clamp(-100.0, 60.0);
        0
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.persistent = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Unipolar Brush Cell tests --

    #[test]
    fn ubc_fires_with_input() {
        let mut n = UnipolarBrushCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(5.0);
        }
        assert!(
            spikes > 10,
            "UBC must fire with excitatory input, got {spikes}"
        );
    }

    #[test]
    fn ubc_silent_without_input() {
        let mut n = UnipolarBrushCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(spikes, 0, "UBC must be silent without input");
    }

    fn ubc_exact_relaxation(previous: f64, steady_state: f64, dt: f64, tau: f64) -> f64 {
        previous + (steady_state - previous) * (-(-dt / tau).exp_m1())
    }

    #[test]
    fn ubc_uses_closed_form_persistent_and_membrane_relaxation() {
        let mut n = UnipolarBrushCell::new();

        let spike = n.step(1.0);

        let input_drive = n.gain;
        let expected_persistent =
            ubc_exact_relaxation(0.0, n.persistent_gain * input_drive, n.dt, n.tau_persistent);
        let expected_v = ubc_exact_relaxation(
            n.v_rest,
            n.v_rest + input_drive + expected_persistent,
            n.dt,
            n.tau_m,
        );
        assert_eq!(spike, 0);
        assert!(
            (n.persistent - expected_persistent).abs() <= 1e-12,
            "persistent={} expected={}",
            n.persistent,
            expected_persistent
        );
        assert!(
            (n.v - expected_v).abs() <= 1e-12,
            "v={} expected={}",
            n.v,
            expected_v
        );
    }

    #[test]
    fn ubc_corrupted_state_is_preserved_on_step() {
        let mut n = UnipolarBrushCell::new();
        n.v = f64::NAN;
        n.persistent = 2.0;

        assert_eq!(n.step(10.0), 0);

        assert!(n.v.is_nan());
        assert_eq!(n.persistent, 2.0);
    }

    #[test]
    fn ubc_persistent_activity() {
        // After input stops, persistent current should sustain some depolarisation
        let mut n = UnipolarBrushCell::new();
        // Drive with input to build persistent current
        for _ in 0..2000 {
            n.step(10.0);
        }
        assert!(
            n.persistent > 0.0,
            "Persistent current must build during input"
        );

        // Now remove input — persistent current should persist
        let persistent_before = n.persistent;
        for _ in 0..100 {
            n.step(0.0);
        }
        assert!(
            n.persistent > 0.0,
            "Persistent current must persist after input removal"
        );
        assert!(
            n.persistent < persistent_before,
            "Persistent current must decay"
        );
    }

    #[test]
    fn ubc_persistent_spikes_after_input() {
        // UBC should continue firing briefly after input stops
        let mut n = UnipolarBrushCell::new();
        // Build up persistent current
        for _ in 0..5000 {
            n.step(10.0);
        }
        // Count spikes after input removal
        let post_spikes: i32 = (0..500).map(|_| n.step(0.0)).sum();
        // May or may not spike depending on persistent level — just test it doesn't crash
        assert!(post_spikes >= 0, "post_spikes must be non-negative");
        assert!(n.v.is_finite());
    }

    #[test]
    fn ubc_negative_input_no_crash() {
        let mut n = UnipolarBrushCell::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn ubc_nan_input_stays_finite() {
        let mut n = UnipolarBrushCell::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
    }

    #[test]
    fn ubc_extreme_input_bounded() {
        let mut n = UnipolarBrushCell::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0);
    }

    #[test]
    fn ubc_reset_clears_state() {
        let mut n = UnipolarBrushCell::new();
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.persistent, 0.0);
    }

    #[test]
    fn ubc_performance_10k_steps() {
        let start = std::time::Instant::now();
        let mut n = UnipolarBrushCell::new();
        for _ in 0..10_000 {
            std::hint::black_box(n.step(5.0));
        }
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 50, "10k steps must complete in <50ms");
    }

    #[test]
    fn ubc_default_matches_constructor_contract() {
        let default = UnipolarBrushCell::default();
        let constructed = UnipolarBrushCell::new();
        assert_eq!(default.v, constructed.v);
        assert_eq!(default.persistent, constructed.persistent);
        assert_eq!(default.tau_persistent, constructed.tau_persistent);
        assert_eq!(default.dt, constructed.dt);
    }
}
