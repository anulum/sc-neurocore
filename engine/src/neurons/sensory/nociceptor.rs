// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sensory Neuron Models

// ═══════════════════════════════════════════════════════════════════
// Nociceptor — pain receptor
// ═══════════════════════════════════════════════════════════════════

/// Nociceptor — high-threshold pain receptor neuron.
///
/// Only fires above noxious threshold. Sensitisation: repeated
/// stimulation lowers threshold (hyperalgesia). TTX-resistant Na+
/// channels provide slow, broad APs.
///
/// Based on Basbaum et al. 2009 / Gold & Gebhart 2010.
#[derive(Clone, Debug)]
pub struct Nociceptor {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau: f64,
    pub sensitisation: f64, // Threshold reduction (mV)
    pub tau_sens: f64,      // Sensitisation decay (ms)
    pub sens_rate: f64,     // Sensitisation buildup rate
    pub gain: f64,
    pub dt: f64,
}

impl Nociceptor {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            v_rest: -65.0,
            v_reset: -68.0,
            v_threshold: -30.0, // High threshold
            tau: 8.0,
            sensitisation: 0.0,
            tau_sens: 5000.0, // Very slow decay (seconds)
            sens_rate: 0.5,
            gain: 1.0,
            dt: 0.5,
        }
    }

    fn exact_relax(value: f64, target: f64, tau: f64, dt: f64) -> f64 {
        target + (value - target) * (-dt / tau).exp()
    }

    fn biological_voltage(value: f64) -> bool {
        value.is_finite() && (-100.0..=60.0).contains(&value)
    }

    fn is_valid(&self) -> bool {
        Self::biological_voltage(self.v)
            && Self::biological_voltage(self.v_rest)
            && Self::biological_voltage(self.v_reset)
            && Self::biological_voltage(self.v_threshold)
            && self.tau.is_finite()
            && self.tau > 0.0
            && self.sensitisation.is_finite()
            && (0.0..=10.0).contains(&self.sensitisation)
            && self.tau_sens.is_finite()
            && self.tau_sens > 0.0
            && self.sens_rate.is_finite()
            && self.sens_rate >= 0.0
            && self.gain.is_finite()
            && self.gain >= 0.0
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.v_threshold > self.v_reset
            && self.v_threshold > self.v_rest
    }

    /// Step with noxious stimulus intensity (≥ 0). Returns spike (1/0).
    pub fn step(&mut self, stimulus: f64) -> i32 {
        if !self.is_valid() || !stimulus.is_finite() {
            return 0;
        }

        let drive = self.gain * stimulus.max(0.0);
        let v_next = Self::exact_relax(self.v, self.v_rest + drive, self.tau, self.dt);
        if !drive.is_finite() || !v_next.is_finite() {
            return 0;
        }

        let effective_threshold = self.v_threshold - self.sensitisation;
        if v_next >= effective_threshold {
            self.v = self.v_reset;
            // Spike causes sensitisation buildup (capped at 10 mV)
            self.sensitisation = (self.sensitisation + self.sens_rate).min(10.0);
            1
        } else {
            // Sensitisation slowly decays
            let sensitisation_next =
                Self::exact_relax(self.sensitisation, 0.0, self.tau_sens, self.dt).max(0.0);
            if !sensitisation_next.is_finite() {
                return 0;
            }
            self.v = v_next.clamp(-100.0, 60.0);
            self.sensitisation = sensitisation_next;
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.sensitisation = 0.0;
    }
}

impl Default for Nociceptor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nociceptor_high_threshold() {
        let mut n = Nociceptor::new();
        // Sub-threshold
        let low: i32 = (0..500).map(|_| n.step(5.0)).sum();
        assert_eq!(low, 0, "nociceptor should not fire at low stimulus");
        // Supra-threshold
        n.reset();
        let high: i32 = (0..500).map(|_| n.step(50.0)).sum();
        assert!(high > 0, "nociceptor should fire at high stimulus");
    }

    #[test]
    fn nociceptor_closed_form_membrane_and_sensitisation_decay() {
        let mut n = Nociceptor::new();
        n.v = -60.0;
        n.sensitisation = 4.0;
        n.gain = 0.5;

        let stimulus = 8.0;
        let drive = n.gain * stimulus;
        let expected_v = exact_relax_nociceptor(n.v, n.v_rest + drive, n.tau, n.dt);
        let expected_sensitisation =
            exact_relax_nociceptor(n.sensitisation, 0.0, n.tau_sens, n.dt).max(0.0);

        assert_eq!(n.step(stimulus), 0);
        assert_close_nociceptor(n.v, expected_v, 1e-12);
        assert_close_nociceptor(n.sensitisation, expected_sensitisation, 1e-12);
    }

    #[test]
    fn nociceptor_invalid_input_preserves_state() {
        let mut n = Nociceptor::new();
        n.v = -60.0;
        n.sensitisation = 2.0;
        let before = n.clone();

        assert_eq!(n.step(f64::NAN), 0);
        assert_eq!(n.v, before.v);
        assert_eq!(n.sensitisation, before.sensitisation);
    }

    #[test]
    fn nociceptor_corrupted_state_preserved_on_step() {
        let mut n = Nociceptor::new();
        n.sensitisation = f64::NAN;
        let before = n.clone();

        assert_eq!(n.step(50.0), 0);
        assert_eq!(n.v, before.v);
        assert!(n.sensitisation.is_nan());
    }

    #[test]
    fn nociceptor_invalid_voltage_preserved_on_step() {
        let mut n = Nociceptor::new();
        n.v = -100.1;
        let before = n.clone();

        assert_eq!(n.step(50.0), 0);
        assert_eq!(n.v, before.v);
        assert_eq!(n.sensitisation, before.sensitisation);
    }

    #[test]
    fn nociceptor_overflowing_drive_preserves_state() {
        let mut n = Nociceptor::new();
        n.gain = f64::MAX;
        let before = n.clone();

        assert_eq!(n.step(2.0), 0);
        assert_eq!(n.v, before.v);
        assert_eq!(n.sensitisation, before.sensitisation);
    }

    fn exact_relax_nociceptor(value: f64, target: f64, tau: f64, dt: f64) -> f64 {
        target + (value - target) * (-dt / tau).exp()
    }

    fn assert_close_nociceptor(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={:.16e} expected={:.16e} tolerance={:.3e}",
            actual,
            expected,
            tolerance
        );
    }

    #[test]
    fn nociceptor_sensitisation() {
        let mut n = Nociceptor::new();
        // Strong stimulus → spikes → sensitisation builds
        for _ in 0..1000 {
            n.step(50.0);
        }
        assert!(n.sensitisation > 0.0, "sensitisation should increase");
        let sens = n.sensitisation;
        // After a long pause, sensitisation decays (tau_sens=5000ms, need many steps)
        for _ in 0..50000 {
            n.step(0.0);
        }
        assert!(
            n.sensitisation < sens,
            "sensitisation should decay: was {sens}, now {}",
            n.sensitisation
        );
    }

    #[test]
    fn nociceptor_no_fire_without_stimulus() {
        let mut n = Nociceptor::new();
        let spikes: i32 = (0..1000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn nociceptor_reset() {
        let mut n = Nociceptor::new();
        for _ in 0..500 {
            n.step(50.0);
        }
        n.reset();
        assert_eq!(n.sensitisation, 0.0);
    }

    #[test]
    fn nociceptor_default_matches_constructor_contract() {
        let default = Nociceptor::default();
        let constructed = Nociceptor::new();
        assert_eq!(default.v, constructed.v);
        assert_eq!(default.sensitisation, constructed.sensitisation);
        assert_eq!(default.v_threshold, constructed.v_threshold);
        assert_eq!(default.dt, constructed.dt);
    }
}
