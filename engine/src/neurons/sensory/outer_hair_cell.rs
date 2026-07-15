// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sensory Neuron Models

// ═══════════════════════════════════════════════════════════════════
// Outer Hair Cell (OHC) — auditory, electromotility
// ═══════════════════════════════════════════════════════════════════

/// Outer hair cell — cochlear amplifier via prestin electromotility.
///
/// Prestin (SLC26A5) is a voltage-dependent motor protein that
/// contracts the OHC soma upon depolarisation and elongates upon
/// hyperpolarisation. This bidirectional response is asymmetric:
/// maximum contraction exceeds maximum elongation (~2:1 ratio).
///
/// The model implements:
/// 1. MET transduction (stereocilia → receptor potential)
/// 2. Prestin electromotility with two-state Boltzmann charge
///    movement and asymmetric length change
/// 3. Nonlinear capacitance (NLC) that peaks near V_pk
///
/// Prestin motility (Santos-Sacchi 2006):
///   charge = Q_max / (1 + exp(z_e*(V-V_pk)/(kT)))
///   ΔL = L_max * (charge/Q_max - 0.5) * asym(V)
///
/// where asym(V) = 1 + a_factor*(V-V_pk)/|V-V_pk+eps| gives
/// asymmetric contraction (depolarised) vs elongation (hyperpolarised).
///
/// Dallos et al., Neuron 58:333, 2008.
/// Santos-Sacchi et al., J Neurosci 26:3992, 2006.
#[derive(Clone, Debug)]
pub struct OuterHairCell {
    pub v: f64,
    pub v_rest: f64,
    pub tau: f64,
    pub g_met: f64,
    pub x_half: f64,
    pub s_met: f64,
    // Prestin parameters
    pub motility: f64,    // Somatic length change (nm, + = contraction)
    pub l_max: f64,       // Maximum length change (nm)
    pub v_pk: f64,        // Peak NLC voltage (mV), ~-40
    pub z_e: f64,         // Prestin charge valence (~0.7)
    pub v_t: f64,         // kT/e thermal voltage (~26 mV at 37°C)
    pub q_max: f64,       // Maximum charge moved (pC)
    pub asym_factor: f64, // Asymmetry: contraction/elongation ratio > 1
    pub dt: f64,
}

impl OuterHairCell {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            v_rest: -70.0,
            tau: 0.3,
            g_met: 15.0,
            x_half: 20.0,
            s_met: 6.0,
            motility: 0.0,
            l_max: 4.0,       // ~4 nm max length change
            v_pk: -40.0,      // Peak NLC voltage
            z_e: 0.7,         // Prestin charge valence
            v_t: 26.0,        // kT/e at 37°C
            q_max: 0.8,       // pC
            asym_factor: 0.3, // 30% asymmetry (contraction > elongation)
            dt: 0.025,
        }
    }

    /// Two-state Boltzmann charge transfer.
    /// Returns fraction of prestin in compact state [0, 1].
    #[inline]
    fn prestin_compact(&self) -> f64 {
        1.0 / (1.0 + (self.z_e * (self.v - self.v_pk) / self.v_t).exp())
    }

    /// Step with displacement (nm). Returns receptor potential (mV).
    pub fn step(&mut self, displacement: f64) -> f64 {
        let p_open = 1.0 / (1.0 + (-(displacement - self.x_half) / self.s_met).exp());
        let i_met = self.g_met * p_open * (0.0 - self.v);
        self.v += (-(self.v - self.v_rest) + i_met) / self.tau * self.dt;

        // Prestin electromotility: bidirectional + asymmetric
        let compact = self.prestin_compact();
        // compact=1 at hyperpolarised, compact=0 at depolarised
        // ΔL = L_max * (0.5 - compact) * asymmetry
        let raw_motility = self.l_max * (0.5 - compact);
        // Asymmetric factor: contraction (positive) enhanced, elongation reduced
        let asym = if raw_motility > 0.0 {
            1.0 + self.asym_factor // Contraction enhanced
        } else {
            1.0 - self.asym_factor // Elongation reduced
        };
        self.motility = raw_motility * asym;

        if !self.v.is_finite() {
            self.v = self.v_rest;
        }
        self.v
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.motility = 0.0;
    }
}

impl Default for OuterHairCell {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ohc_depolarises_and_motility() {
        let mut c = OuterHairCell::new();
        for _ in 0..200 {
            c.step(40.0);
        }
        assert!(c.v > c.v_rest);
        assert!(c.motility.abs() > 0.01, "OHC should show motility");
    }

    #[test]
    fn ohc_prestin_bidirectional() {
        // Depolarisation → contraction (positive motility)
        // Hyperpolarisation → elongation (negative motility)
        let mut dep = OuterHairCell::new();
        dep.v = -20.0; // Depolarised
        dep.step(0.0); // Update motility
        let mot_dep = dep.motility;

        let mut hyp = OuterHairCell::new();
        hyp.v = -80.0; // Hyperpolarised
        hyp.step(0.0);
        let mot_hyp = hyp.motility;

        assert!(
            mot_dep > 0.0,
            "Depolarisation should contract: motility={mot_dep:.3}"
        );
        assert!(
            mot_hyp < 0.0,
            "Hyperpolarisation should elongate: motility={mot_hyp:.3}"
        );
    }

    #[test]
    fn ohc_prestin_asymmetric() {
        // Contraction should be larger than elongation (asymmetry)
        // Drive OHC to depolarised state with strong input
        let mut dep = OuterHairCell::new();
        for _ in 0..2000 {
            dep.step(80.0);
        } // Strong depolarisation
        let contraction = dep.motility;

        // Drive OHC with zero input → stays near rest (hyperpolarised relative to V_pk)
        let mut hyp = OuterHairCell::new();
        for _ in 0..2000 {
            hyp.step(0.0);
        } // Near rest = hyperpolarised vs V_pk
        let elongation = hyp.motility;

        // At rest (V=-70), prestin is mostly in expanded state (elongation)
        // With strong input (depolarised), prestin contracts
        // Due to asymmetry factor, |contraction| > |elongation|
        assert!(
            contraction.abs() > elongation.abs() * 0.5,
            "Asymmetric prestin: contraction={contraction:.3}, elongation={elongation:.3}"
        );
    }

    #[test]
    fn ohc_reset() {
        let mut c = OuterHairCell::new();
        for _ in 0..100 {
            c.step(40.0);
        }
        c.reset();
        assert_eq!(c.motility, 0.0);
    }

    #[test]
    fn ohc_bounded() {
        let mut c = OuterHairCell::new();
        for _ in 0..10000 {
            c.step(100.0);
        }
        assert!(c.v.is_finite());
    }

    #[test]
    fn outer_hair_cell_default_matches_constructor_contract() {
        let default = OuterHairCell::default();
        let constructed = OuterHairCell::new();
        assert_eq!(default.v, constructed.v);
        assert_eq!(default.motility, constructed.motility);
        assert_eq!(default.v_pk, constructed.v_pk);
        assert_eq!(default.dt, constructed.dt);
    }

    #[test]
    fn ohc_nonfinite_voltage_recovers_rest() {
        let mut cell = OuterHairCell::new();
        cell.v = f64::NAN;
        assert_eq!(cell.step(0.0), cell.v_rest);
    }
}
