// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sensory Neuron Models

// ═══════════════════════════════════════════════════════════════════
// Pacinian Corpuscle — rapidly adapting mechanoreceptor
// ═══════════════════════════════════════════════════════════════════

/// Pacinian corpuscle — rapidly adapting (RA/RAII) mechanoreceptor.
///
/// Responds to vibration and transient pressure changes.
/// Band-pass filtering via lamellar structure: only signals
/// with rapid onset/offset produce responses. Derivative-like.
///
/// Based on Loewenstein & Skalak 1966 / Bell et al. 1994.
#[derive(Clone, Debug)]
pub struct PacinianCorpuscle {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau: f64,
    pub prev_pressure: f64,
    pub adapt: f64,
    pub tau_adapt: f64,
    pub gain: f64,
    pub dt: f64,
}

impl PacinianCorpuscle {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            v_rest: -65.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau: 2.0,
            prev_pressure: 0.0,
            adapt: 0.0,
            tau_adapt: 5.0, // Fast adaptation
            gain: 10.0,     // High gain on derivative
            dt: 0.5,
        }
    }

    #[inline]
    fn exact_relax(value: f64, target: f64, tau: f64, dt: f64) -> f64 {
        target + (value - target) * (-dt / tau).exp()
    }

    fn is_valid(&self) -> bool {
        [
            self.v,
            self.v_rest,
            self.v_reset,
            self.v_threshold,
            self.tau,
            self.prev_pressure,
            self.adapt,
            self.tau_adapt,
            self.gain,
            self.dt,
        ]
        .iter()
        .all(|value| value.is_finite())
            && (-100.0..=60.0).contains(&self.v)
            && self.tau > 0.0
            && self.tau_adapt > 0.0
            && self.gain >= 0.0
            && self.dt > 0.0
            && self.adapt >= 0.0
            && self.v_threshold > self.v_reset
            && self.v_threshold > self.v_rest
    }

    /// Step with pressure (arbitrary units). Returns spike (1/0).
    pub fn step(&mut self, pressure: f64) -> i32 {
        if !self.is_valid() || !pressure.is_finite() {
            return 0;
        }

        // Derivative-like response: rate of change drives the neuron
        let dp = (pressure - self.prev_pressure) / self.dt;
        let drive = self.gain * dp.abs() - self.adapt;
        let v_inf = self.v_rest + drive;
        let v_next = Self::exact_relax(self.v, v_inf, self.tau, self.dt);
        let adapt_inf = 0.5 * drive.max(0.0);
        let adapt_next = Self::exact_relax(self.adapt, adapt_inf, self.tau_adapt, self.dt).max(0.0);
        if !dp.is_finite() || !drive.is_finite() || !v_next.is_finite() || !adapt_next.is_finite() {
            return 0;
        }

        self.prev_pressure = pressure;
        self.adapt = adapt_next;
        if v_next >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            self.v = v_next.clamp(-100.0, 60.0);
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.prev_pressure = 0.0;
        self.adapt = 0.0;
    }
}

impl Default for PacinianCorpuscle {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pacinian_fires_on_pressure_onset() {
        let mut p = PacinianCorpuscle::new();
        // Ramp up pressure rapidly
        let spikes: i32 = (0..100).map(|i| p.step(i as f64 * 2.0)).sum();
        assert!(spikes > 0, "Pacinian should fire on pressure onset");
    }

    #[test]
    fn pacinian_adapts_to_sustained() {
        let mut p = PacinianCorpuscle::new();
        // Rapid onset
        let onset: i32 = (0..10).map(|i| p.step(i as f64 * 10.0)).sum();
        // Sustained (constant pressure, dp/dt ≈ 0)
        let sustained: i32 = (0..500).map(|_| p.step(100.0)).sum();
        // Should fire mostly during onset, not during sustained
        assert!(
            sustained <= onset + 5,
            "Pacinian should adapt to sustained: onset={onset}, sustained={sustained}"
        );
    }

    #[test]
    fn pacinian_no_fire_at_rest() {
        let mut p = PacinianCorpuscle::new();
        let spikes: i32 = (0..500).map(|_| p.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn pacinian_closed_form_membrane_and_adaptation_relaxation() {
        let mut p = PacinianCorpuscle::new();
        p.v = -66.0;
        p.prev_pressure = 5.0;
        p.adapt = 0.2;
        p.gain = 0.0;

        let pressure = 5.0;
        let dp = (pressure - p.prev_pressure) / p.dt;
        let drive = p.gain * dp.abs() - p.adapt;
        let expected_v = exact_relax_pacinian(p.v, p.v_rest + drive, p.tau, p.dt);
        let expected_adapt =
            exact_relax_pacinian(p.adapt, 0.5 * drive.max(0.0), p.tau_adapt, p.dt).max(0.0);

        assert_eq!(p.step(pressure), 0);
        assert_eq!(p.prev_pressure, pressure);
        assert_close_pacinian(p.v, expected_v, 1e-12);
        assert_close_pacinian(p.adapt, expected_adapt, 1e-12);
    }

    #[test]
    fn pacinian_invalid_input_preserves_state() {
        let mut p = PacinianCorpuscle::new();
        p.prev_pressure = 12.0;
        p.adapt = 0.4;
        let before = p.clone();
        assert_eq!(p.step(f64::NAN), 0);
        assert_eq!(p.v, before.v);
        assert_eq!(p.prev_pressure, before.prev_pressure);
        assert_eq!(p.adapt, before.adapt);
    }

    #[test]
    fn pacinian_corrupted_state_preserved_on_step() {
        let mut p = PacinianCorpuscle::new();
        p.prev_pressure = f64::NAN;
        let before = p.clone();
        assert_eq!(p.step(10.0), 0);
        assert_eq!(p.v, before.v);
        assert!(p.prev_pressure.is_nan());
        assert_eq!(p.adapt, before.adapt);
    }

    #[test]
    fn pacinian_invalid_voltage_preserved_on_step() {
        let mut p = PacinianCorpuscle::new();
        p.v = -100.1;
        let before = p.clone();
        assert_eq!(p.step(10.0), 0);
        assert_eq!(p.v, before.v);
        assert_eq!(p.prev_pressure, before.prev_pressure);
        assert_eq!(p.adapt, before.adapt);
    }

    fn exact_relax_pacinian(value: f64, target: f64, tau: f64, dt: f64) -> f64 {
        target + (value - target) * (-dt / tau).exp()
    }

    fn assert_close_pacinian(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={:.16e} expected={:.16e} tolerance={:.3e}",
            actual,
            expected,
            tolerance
        );
    }

    #[test]
    fn pacinian_reset() {
        let mut p = PacinianCorpuscle::new();
        for i in 0..100 {
            p.step(i as f64);
        }
        p.reset();
        assert_eq!(p.prev_pressure, 0.0);
        assert_eq!(p.adapt, 0.0);
    }

    #[test]
    fn pacinian_default_matches_constructor_contract() {
        let default = PacinianCorpuscle::default();
        let constructed = PacinianCorpuscle::new();
        assert_eq!(default.v, constructed.v);
        assert_eq!(default.prev_pressure, constructed.prev_pressure);
        assert_eq!(default.adapt, constructed.adapt);
        assert_eq!(default.dt, constructed.dt);
    }

    #[test]
    fn pacinian_overflowing_candidate_preserves_state() {
        let mut cell = PacinianCorpuscle::new();
        cell.gain = f64::MAX;
        let before = (cell.v, cell.prev_pressure, cell.adapt);
        assert_eq!(cell.step(2.0), 0);
        assert_eq!((cell.v, cell.prev_pressure, cell.adapt), before);
    }
}
