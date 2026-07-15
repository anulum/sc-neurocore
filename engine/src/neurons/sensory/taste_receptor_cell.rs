// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sensory Neuron Models

// ═══════════════════════════════════════════════════════════════════
// Taste Receptor Cell
// ═══════════════════════════════════════════════════════════════════

/// Taste receptor cell — gustatory transducer.
///
/// Type II cells: GPCR → PLC → IP3 → Ca2+ release → ATP secretion.
/// Graded output (ATP release proportional to Ca2+), no conventional
/// spikes. Adapts via Ca2+ pump.
///
/// Based on Chaudhari & Roper 2010 / Liman et al. 2014.
#[derive(Clone, Debug)]
pub struct TasteReceptorCell {
    pub v: f64,
    pub v_rest: f64,
    pub tau: f64,
    pub ca: f64,  // Intracellular Ca2+ (normalised)
    pub ip3: f64, // IP3 concentration (normalised)
    pub tau_ip3: f64,
    pub tau_ca: f64,
    pub gain: f64,
    pub atp_release: f64, // Output: ATP release rate
    pub dt: f64,
}

impl TasteReceptorCell {
    pub fn new() -> Self {
        Self {
            v: -50.0,
            v_rest: -50.0,
            tau: 10.0,
            ca: 0.0,
            ip3: 0.0,
            tau_ip3: 100.0,
            tau_ca: 200.0,
            gain: 1.0,
            atp_release: 0.0,
            dt: 0.5,
        }
    }

    /// Step with tastant concentration (≥ 0). Returns receptor potential (mV).
    pub fn step(&mut self, tastant: f64) -> f64 {
        let conc = tastant.max(0.0);
        // GPCR → IP3
        let ip3_target = conc / (conc + 0.5);
        self.ip3 += (ip3_target - self.ip3) / self.tau_ip3 * self.dt;
        self.ip3 = self.ip3.clamp(0.0, 1.0);

        // IP3 → Ca2+ release from ER
        let ca_release = self.ip3.powi(2) * (1.0 - self.ca);
        self.ca += (ca_release - self.ca / self.tau_ca) * self.dt;
        self.ca = self.ca.clamp(0.0, 1.0);

        // Ca2+ → depolarisation (TRPM5 channel)
        let i_trpm5 = self.gain * self.ca * 20.0;
        self.v += (-(self.v - self.v_rest) + i_trpm5) / self.tau * self.dt;

        // ATP release proportional to Ca2+
        self.atp_release = self.ca;

        self.v
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.ca = 0.0;
        self.ip3 = 0.0;
        self.atp_release = 0.0;
    }
}

impl Default for TasteReceptorCell {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taste_depolarises_with_tastant() {
        let mut t = TasteReceptorCell::new();
        let v_rest = t.v;
        for _ in 0..500 {
            t.step(5.0);
        }
        assert!(t.v > v_rest, "taste cell should depolarise");
    }

    #[test]
    fn taste_atp_release() {
        let mut t = TasteReceptorCell::new();
        for _ in 0..500 {
            t.step(5.0);
        }
        assert!(t.atp_release > 0.0, "ATP should be released");
    }

    #[test]
    fn taste_no_response_without_tastant() {
        let mut t = TasteReceptorCell::new();
        for _ in 0..500 {
            t.step(0.0);
        }
        assert!((t.v - t.v_rest).abs() < 2.0);
        assert!(t.atp_release < 0.01);
    }

    #[test]
    fn taste_ca_bounded() {
        let mut t = TasteReceptorCell::new();
        for _ in 0..10000 {
            t.step(100.0);
        }
        assert!(t.ca >= 0.0 && t.ca <= 1.0);
        assert!(t.ip3 >= 0.0 && t.ip3 <= 1.0);
    }

    #[test]
    fn taste_reset() {
        let mut t = TasteReceptorCell::new();
        for _ in 0..500 {
            t.step(5.0);
        }
        t.reset();
        assert_eq!(t.ca, 0.0);
        assert_eq!(t.ip3, 0.0);
        assert_eq!(t.atp_release, 0.0);
    }

    #[test]
    fn taste_receptor_default_matches_constructor_contract() {
        let default = TasteReceptorCell::default();
        let constructed = TasteReceptorCell::new();
        assert_eq!(default.v, constructed.v);
        assert_eq!(default.ca, constructed.ca);
        assert_eq!(default.ip3, constructed.ip3);
        assert_eq!(default.atp_release, constructed.atp_release);
    }
}
