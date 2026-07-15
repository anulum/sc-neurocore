// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sensory Neuron Models

// ═══════════════════════════════════════════════════════════════════
// Merkel Cell — slowly adapting type I mechanoreceptor
// ═══════════════════════════════════════════════════════════════════

/// Merkel cell — slowly adapting type I (SAI) mechanoreceptor.
///
/// Responds to sustained pressure with slowly adapting discharge.
/// Encodes texture and edges. Two-component model: fast onset + slow
/// sustained component.
///
/// Based on Lesniak et al. 2014.
#[derive(Clone, Debug)]
pub struct MerkelCell {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau: f64,
    pub adapt: f64,     // Slow adaptation variable
    pub tau_adapt: f64, // Adaptation time constant (ms)
    pub a_adapt: f64,   // Adaptation coupling
    pub gain: f64,
    pub dt: f64,
}

impl MerkelCell {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            v_rest: -65.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau: 5.0,
            adapt: 0.0,
            tau_adapt: 200.0, // Very slow adaptation
            a_adapt: 0.3,
            gain: 1.5,
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
            self.adapt,
            self.tau_adapt,
            self.a_adapt,
            self.gain,
            self.dt,
        ]
        .iter()
        .all(|value| value.is_finite())
            && (-100.0..=60.0).contains(&self.v)
            && self.tau > 0.0
            && self.tau_adapt > 0.0
            && self.a_adapt >= 0.0
            && self.gain >= 0.0
            && self.dt > 0.0
            && self.adapt >= 0.0
            && self.v_threshold > self.v_reset
            && self.v_threshold > self.v_rest
    }

    /// Step with pressure (arbitrary units, ≥ 0). Returns spike (1/0).
    pub fn step(&mut self, pressure: f64) -> i32 {
        if !self.is_valid() || !pressure.is_finite() {
            return 0;
        }

        let rectified_pressure = pressure.max(0.0);
        let v_inf = self.v_rest + self.gain * rectified_pressure - self.adapt;
        let v_next = Self::exact_relax(self.v, v_inf, self.tau, self.dt);
        let adapt_inf = (self.a_adapt * (v_next - self.v_rest).max(0.0)).max(0.0);
        let adapt_next = Self::exact_relax(self.adapt, adapt_inf, self.tau_adapt, self.dt).max(0.0);
        if !v_next.is_finite() || !adapt_next.is_finite() {
            return 0;
        }

        if v_next >= self.v_threshold {
            self.v = self.v_reset;
            self.adapt = adapt_next;
            1
        } else {
            self.v = v_next.clamp(-100.0, 60.0);
            self.adapt = adapt_next;
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.adapt = 0.0;
    }
}

impl Default for MerkelCell {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merkel_fires_with_sustained_pressure() {
        let mut m = MerkelCell::new();
        let spikes: i32 = (0..2000).map(|_| m.step(20.0)).sum();
        assert!(spikes > 0, "Merkel should fire with sustained pressure");
    }

    #[test]
    fn merkel_slow_adaptation() {
        let mut m = MerkelCell::new();
        let first: i32 = (0..1000).map(|_| m.step(20.0)).sum();
        let second: i32 = (0..1000).map(|_| m.step(20.0)).sum();
        // Slow adapting: second half may fire slightly fewer but still fires
        assert!(
            second > 0,
            "Merkel should still fire in second half (slow adapting)"
        );
        assert!(second <= first + 5, "Merkel should slowly adapt");
    }

    #[test]
    fn merkel_no_fire_without_pressure() {
        let mut m = MerkelCell::new();
        let spikes: i32 = (0..1000).map(|_| m.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn merkel_closed_form_membrane_and_adaptation_relaxation() {
        let mut m = MerkelCell::new();
        m.v = -66.0;
        m.adapt = 0.2;
        m.gain = 0.0;

        let v_inf = m.v_rest - m.adapt;
        let expected_v = exact_relax_merkel(m.v, v_inf, m.tau, m.dt);
        let adapt_inf = (m.a_adapt * (expected_v - m.v_rest).max(0.0)).max(0.0);
        let expected_adapt = exact_relax_merkel(m.adapt, adapt_inf, m.tau_adapt, m.dt).max(0.0);

        assert_eq!(m.step(0.0), 0);
        assert_close_merkel(m.v, expected_v, 1e-12);
        assert_close_merkel(m.adapt, expected_adapt, 1e-12);
    }

    #[test]
    fn merkel_invalid_input_preserves_state() {
        let mut m = MerkelCell::new();
        let before = m.clone();
        assert_eq!(m.step(f64::NAN), 0);
        assert_eq!(m.v, before.v);
        assert_eq!(m.adapt, before.adapt);
    }

    #[test]
    fn merkel_corrupted_state_preserved_on_step() {
        let mut m = MerkelCell::new();
        m.adapt = f64::NAN;
        let before = m.clone();
        assert_eq!(m.step(20.0), 0);
        assert_eq!(m.v, before.v);
        assert!(m.adapt.is_nan());
    }

    #[test]
    fn merkel_invalid_voltage_preserved_on_step() {
        let mut m = MerkelCell::new();
        m.v = 60.1;
        let before = m.clone();
        assert_eq!(m.step(20.0), 0);
        assert_eq!(m.v, before.v);
        assert_eq!(m.adapt, before.adapt);
    }

    fn exact_relax_merkel(value: f64, target: f64, tau: f64, dt: f64) -> f64 {
        target + (value - target) * (-dt / tau).exp()
    }

    fn assert_close_merkel(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={:.16e} expected={:.16e} tolerance={:.3e}",
            actual,
            expected,
            tolerance
        );
    }

    #[test]
    fn merkel_reset() {
        let mut m = MerkelCell::new();
        for _ in 0..500 {
            m.step(20.0);
        }
        m.reset();
        assert_eq!(m.adapt, 0.0);
    }

    #[test]
    fn merkel_default_matches_constructor_contract() {
        let default = MerkelCell::default();
        let constructed = MerkelCell::new();
        assert_eq!(default.v, constructed.v);
        assert_eq!(default.adapt, constructed.adapt);
        assert_eq!(default.tau_adapt, constructed.tau_adapt);
        assert_eq!(default.dt, constructed.dt);
    }

    #[test]
    fn merkel_overflowing_candidate_preserves_state() {
        let mut cell = MerkelCell::new();
        cell.gain = f64::MAX;
        let before = (cell.v, cell.adapt);
        assert_eq!(cell.step(2.0), 0);
        assert_eq!((cell.v, cell.adapt), before);
    }
}
