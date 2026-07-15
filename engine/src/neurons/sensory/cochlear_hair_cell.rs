// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sensory Neuron Models

/// Cochlear inner hair cell: mechano-electrical transduction.
///
/// Converts basilar membrane displacement (mechanical) into receptor potential
/// via stereocilia tip-link channels with Boltzmann activation:
///
///   P_open(x) = 1 / (1 + exp(-(x - x_0) / δ))
///   I_MET = g_max · P_open · (V - E_MET)
///   C dV/dt = -g_L(V - E_L) - I_MET + I_ext
///
/// Reference: Meddis (2006), Zilany et al. (2009, 2014).
#[derive(Clone, Debug)]
pub struct CochlearHairCell {
    pub v: f64,
    pub g_max: f64,
    pub e_met: f64,
    pub g_l: f64,
    pub e_l: f64,
    pub cap: f64,
    pub x0: f64,
    pub delta: f64,
    pub dt: f64,
    pub glutamate_release: f64,
}

impl CochlearHairCell {
    pub fn new() -> Self {
        Self {
            v: -60.0,
            g_max: 10.0,
            e_met: 0.0,
            g_l: 1.0,
            e_l: -60.0,
            cap: 10.0,
            x0: 0.0,
            delta: 0.1,
            dt: 0.01,
            glutamate_release: 0.0,
        }
    }

    /// Boltzmann activation of MET channels.
    fn p_open(&self, displacement: f64) -> f64 {
        let z = (displacement - self.x0) / self.delta;
        if z >= 0.0 {
            1.0 / (1.0 + (-z).exp())
        } else {
            let ez = z.exp();
            ez / (1.0 + ez)
        }
    }

    fn valid_runtime(&self) -> bool {
        [
            self.v,
            self.g_max,
            self.e_met,
            self.g_l,
            self.e_l,
            self.cap,
            self.x0,
            self.delta,
            self.dt,
            self.glutamate_release,
        ]
        .iter()
        .all(|x| x.is_finite())
            && self.g_max >= 0.0
            && self.g_l > 0.0
            && self.cap > 0.0
            && self.delta > 0.0
            && self.dt > 0.0
            && self.glutamate_release >= 0.0
    }

    /// Step with basilar membrane displacement.
    pub fn step(&mut self, displacement: f64) -> i32 {
        if !self.valid_runtime() || !displacement.is_finite() {
            return 0;
        }
        let po = self.p_open(displacement);
        let g_met = self.g_max * po;
        let g_total = self.g_l + g_met;
        if !(g_total.is_finite() && g_total > 0.0) {
            return 0;
        }
        let v_inf = (self.g_l * self.e_l + g_met * self.e_met) / g_total;
        let candidate_v = v_inf + (self.v - v_inf) * (-(g_total / self.cap) * self.dt).exp();
        let candidate_release = (candidate_v + 60.0).max(0.0) / 40.0;
        if !(candidate_v.is_finite() && candidate_release.is_finite()) {
            return 0;
        }
        self.v = candidate_v;

        // Graded glutamate release (no spike, but we return 1 if above threshold).
        self.glutamate_release = candidate_release;
        if self.glutamate_release > 0.5 {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.e_l;
        self.glutamate_release = 0.0;
    }
}

impl Default for CochlearHairCell {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cochlear_displacement_depolarises() {
        let mut cell = CochlearHairCell::new();
        let v_rest = cell.v;
        for _ in 0..100 {
            cell.step(0.5);
        }
        assert!(
            cell.v > v_rest,
            "Positive displacement should depolarise: {:.2} > {:.2}",
            cell.v,
            v_rest
        );
    }

    #[test]
    fn cochlear_matches_closed_form_membrane_relaxation() {
        let mut cell = CochlearHairCell::new();
        let po = 1.0 / (1.0 + (-(0.0 - cell.x0) / cell.delta).exp());
        let g_met = cell.g_max * po;
        let g_total = cell.g_l + g_met;
        let v_inf = (cell.g_l * cell.e_l + g_met * cell.e_met) / g_total;
        let expected = v_inf + (cell.v - v_inf) * (-(g_total / cell.cap) * cell.dt).exp();
        let spike = cell.step(0.0);
        assert!(spike == 0 || spike == 1);
        assert!((cell.v - expected).abs() < 1e-12);
    }

    #[test]
    fn cochlear_invalid_runtime_preserves_state() {
        let mut cell = CochlearHairCell::new();
        cell.v = -55.0;
        cell.glutamate_release = 0.125;
        let before = (cell.v, cell.glutamate_release);
        cell.cap = -1.0;
        assert_eq!(cell.step(0.25), 0);
        assert_eq!((cell.v, cell.glutamate_release), before);
    }

    #[test]
    fn cochlear_graded_release() {
        let mut cell = CochlearHairCell::new();
        for _ in 0..200 {
            cell.step(1.0);
        }
        assert!(
            cell.glutamate_release > 0.0,
            "Should release glutamate: {:.4}",
            cell.glutamate_release
        );
    }

    #[test]
    fn cochlear_zero_displacement_rest() {
        let mut cell = CochlearHairCell::new();
        for _ in 0..100 {
            cell.step(0.0);
        }
        // At P_open(0) = 0.5, steady MET current depolarises V above E_L.
        assert!(
            cell.v > -80.0 && cell.v < 0.0,
            "Zero displacement → physiological range: {:.2}",
            cell.v
        );
    }

    #[test]
    fn cochlear_reset() {
        let mut cell = CochlearHairCell::new();
        for _ in 0..100 {
            cell.step(1.0);
        }
        cell.reset();
        assert_eq!(cell.v, cell.e_l);
        assert_eq!(cell.glutamate_release, 0.0);
    }

    #[test]
    fn cochlear_hair_cell_default_matches_constructor_contract() {
        let default = CochlearHairCell::default();
        let constructed = CochlearHairCell::new();
        assert_eq!(default.v, constructed.v);
        assert_eq!(default.g_max, constructed.g_max);
        assert_eq!(default.delta, constructed.delta);
        assert_eq!(default.glutamate_release, constructed.glutamate_release);
    }

    #[test]
    fn cochlear_negative_displacement_uses_stable_boltzmann_branch() {
        let mut cell = CochlearHairCell::new();
        assert_eq!(cell.step(-1.0), 0);
        assert!(cell.v.is_finite());
        assert!(cell.glutamate_release.is_finite());
    }

    #[test]
    fn cochlear_nonfinite_total_conductance_preserves_state() {
        let mut cell = CochlearHairCell::new();
        cell.g_l = f64::MAX;
        cell.g_max = f64::MAX;
        let before = (cell.v, cell.glutamate_release);
        assert_eq!(cell.step(1.0), 0);
        assert_eq!((cell.v, cell.glutamate_release), before);
    }

    #[test]
    fn cochlear_nonfinite_candidate_preserves_state() {
        let mut cell = CochlearHairCell::new();
        cell.g_l = 2.0;
        cell.g_max = 0.0;
        cell.e_l = f64::MAX;
        let before = (cell.v, cell.glutamate_release);
        assert_eq!(cell.step(0.0), 0);
        assert_eq!((cell.v, cell.glutamate_release), before);
    }
}
