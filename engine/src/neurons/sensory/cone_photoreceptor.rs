// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sensory Neuron Models

// ═══════════════════════════════════════════════════════════════════
// Cone Photoreceptor — photopic vision
// ═══════════════════════════════════════════════════════════════════

/// Cone photoreceptor — photopic (bright light) colour vision.
///
/// Same transduction cascade as rods but faster kinetics, lower
/// sensitivity, and faster dark adaptation.
///
/// Based on Schnapf et al. 1990 / Baylor 1987.
#[derive(Clone, Debug)]
pub struct ConePhotoreceptor {
    pub v: f64,
    pub v_dark: f64,
    pub v_hyper: f64,
    pub cgmp: f64,
    pub tau_act: f64,
    pub tau_rec: f64,
    pub sensitivity: f64,
    pub dt: f64,
}

impl ConePhotoreceptor {
    pub fn new() -> Self {
        Self {
            v: -40.0,
            v_dark: -40.0,
            v_hyper: -65.0,
            cgmp: 1.0,
            tau_act: 5.0,       // Faster than rods
            tau_rec: 50.0,      // Much faster recovery than rods
            sensitivity: 0.001, // Lower sensitivity than rods
            dt: 0.1,
        }
    }

    pub fn step(&mut self, light: f64) -> f64 {
        let light_clamped = light.max(0.0);
        let d_cgmp = -self.sensitivity * light_clamped * self.cgmp / self.tau_act
            + (1.0 - self.cgmp) / self.tau_rec;
        self.cgmp += d_cgmp * self.dt;
        self.cgmp = self.cgmp.clamp(0.0, 1.0);

        let cng_fraction = self.cgmp.powi(3);
        self.v = self.v_hyper + (self.v_dark - self.v_hyper) * cng_fraction;
        self.v
    }

    pub fn reset(&mut self) {
        self.v = self.v_dark;
        self.cgmp = 1.0;
    }
}

impl Default for ConePhotoreceptor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::RodPhotoreceptor;
    use super::*;

    #[test]
    fn cone_hyperpolarises_with_light() {
        let mut c = ConePhotoreceptor::new();
        let v_dark = c.v;
        for _ in 0..500 {
            c.step(500.0);
        }
        assert!(c.v < v_dark);
    }

    #[test]
    fn cone_faster_than_rod() {
        let mut rod = RodPhotoreceptor::new();
        let mut cone = ConePhotoreceptor::new();
        // Flash, then dark
        for _ in 0..500 {
            rod.step(100.0);
            cone.step(100.0);
        }
        for _ in 0..2000 {
            rod.step(0.0);
            cone.step(0.0);
        }
        // Cone should recover more (faster tau_rec)
        let rod_recovery = rod.v - rod.v_hyper;
        let cone_recovery = cone.v - cone.v_hyper;
        assert!(
            cone_recovery > rod_recovery,
            "cone ({cone_recovery:.1}) should recover more than rod ({rod_recovery:.1})"
        );
    }

    #[test]
    fn cone_reset() {
        let mut c = ConePhotoreceptor::new();
        for _ in 0..500 {
            c.step(500.0);
        }
        c.reset();
        assert_eq!(c.cgmp, 1.0);
        assert_eq!(c.v, c.v_dark);
    }

    #[test]
    fn cone_photoreceptor_default_matches_constructor_contract() {
        let default = ConePhotoreceptor::default();
        let constructed = ConePhotoreceptor::new();
        assert_eq!(default.v, constructed.v);
        assert_eq!(default.cgmp, constructed.cgmp);
        assert_eq!(default.tau_rec, constructed.tau_rec);
        assert_eq!(default.dt, constructed.dt);
    }
}
