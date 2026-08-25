// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety mirror for Benda-Herz adaptation
//! Safety reference for the source-faithful Benda–Herz adaptation model.

/// Source state and parameters for equations (8) and (45).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BendaHerzNeuron {
    pub a: f64,
    pub phase: f64,
    pub onset_gain: f64,
    pub rheobase: f64,
    pub adaptation_slope: f64,
    pub tau_a: f64,
    pub dt: f64,
}

impl Default for BendaHerzNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl BendaHerzNeuron {
    /// Construct the published square-root, linear-adaptation example.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            a: 0.0,
            phase: 0.0,
            onset_gain: 60.0,
            rheobase: 0.0,
            adaptation_slope: 0.1,
            tau_a: 100.0,
            dt: 0.1,
        }
    }

    fn rhs(&self, a: f64, current: f64) -> (f64, f64) {
        let rate = self.onset_gain * (current - a - self.rheobase).max(0.0).sqrt();
        (
            (self.adaptation_slope * rate - a) / self.tau_a,
            rate / 1000.0,
        )
    }

    /// Candidate-first RK4 step. `Err` leaves the state unchanged.
    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !self.valid() || !current.is_finite() {
            return Err("invalid Benda-Herz input");
        }
        let (k1a, k1p) = self.rhs(self.a, current);
        let (k2a, k2p) = self.rhs(self.a + 0.5 * self.dt * k1a, current);
        let (k3a, k3p) = self.rhs(self.a + 0.5 * self.dt * k2a, current);
        let (k4a, k4p) = self.rhs(self.a + self.dt * k3a, current);
        let scale = self.dt / 6.0;
        let next_a = self.a + scale * (k1a + 2.0 * k2a + 2.0 * k3a + k4a);
        let next_phase = self.phase + scale * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
        if !next_a.is_finite()
            || next_a < 0.0
            || !next_phase.is_finite()
            || !(0.0..2.0).contains(&next_phase)
        {
            return Err("invalid Benda-Herz candidate");
        }
        self.a = next_a;
        if next_phase >= 1.0 {
            self.phase = 0.0;
            Ok(1)
        } else {
            self.phase = next_phase;
            Ok(0)
        }
    }

    /// Reset all dynamic state.
    pub fn reset(&mut self) {
        self.a = 0.0;
        self.phase = 0.0;
    }

    /// Check the complete state/parameter domain.
    #[must_use]
    pub fn valid(&self) -> bool {
        self.a.is_finite()
            && self.a >= 0.0
            && self.phase.is_finite()
            && (0.0..1.0).contains(&self.phase)
            && self.onset_gain.is_finite()
            && self.onset_gain > 0.0
            && self.rheobase.is_finite()
            && self.adaptation_slope.is_finite()
            && self.adaptation_slope >= 0.0
            && self.tau_a.is_finite()
            && self.tau_a > 0.0
            && self.dt.is_finite()
            && self.dt > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn defaults_are_valid() {
        assert!(BendaHerzNeuron::new().valid());
    }
    #[test]
    fn source_phase_spikes() {
        let mut n = BendaHerzNeuron {
            phase: 0.99,
            dt: 1.0,
            adaptation_slope: 0.0,
            ..BendaHerzNeuron::new()
        };
        assert_eq!(n.step(1.0), Ok(1));
        assert_eq!(n.phase, 0.0);
    }
    #[test]
    fn invalid_is_atomic() {
        let mut n = BendaHerzNeuron::new();
        let before = n;
        assert!(n.step(f64::NAN).is_err());
        assert_eq!(n, before);
    }
}
