// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Cardiac Purkinje Fibre Model

//! DiFrancesco-Noble cardiac Purkinje fibre dynamics.

// ═══════════════════════════════════════════════════════════════════
// Cardiac Purkinje Fibre
// ═══════════════════════════════════════════════════════════════════

/// Cardiac Purkinje fibre — DiFrancesco-Noble 1985 model.
///
/// Specialised cardiac conduction cell with long action potentials
/// (~300 ms) and pacemaker capability via If (funny/HCN current).
///
/// 6 major ionic currents:
/// - **INa** (fast Na, m³h): rapid depolarisation (phase 0)
/// - **ICaL** (L-type Ca²⁺, d·f): plateau maintenance (phase 2)
/// - **IKr** (rapid delayed rectifier K, x_r): phase 3 repolarisation
/// - **IK1** (inward rectifier K): resting potential stabilisation
/// - **If** (funny current, HCN, y): pacemaker depolarisation (phase 4)
/// - **IL** (leak)
///
/// Action potential phases:
/// 0 — rapid depolarisation (INa)
/// 1 — early repolarisation notch
/// 2 — plateau (ICaL vs IKr balance)
/// 3 — repolarisation (IKr dominates)
/// 4 — pacemaker depolarisation (If)
///
/// Uses 10 sub-steps (dt_sub = 0.05 ms) for Na gating stability.
///
/// DiFrancesco & Noble, Phil Trans R Soc Lond B 307:353, 1985.
/// Noble, J Physiol 353:1, 1984 (review).
#[derive(Clone, Debug)]
pub struct CardiacPurkinjeFibre {
    pub v: f64,
    pub m: f64,   // Na activation
    pub h: f64,   // Na inactivation
    pub d: f64,   // CaL activation
    pub f: f64,   // CaL inactivation
    pub x_r: f64, // IKr activation
    pub y: f64,   // If (HCN) activation
    pub c_m: f64,
    pub g_na: f64,
    pub g_cal: f64,
    pub g_kr: f64,
    pub g_k1: f64,
    pub g_f: f64, // Funny current conductance
    pub g_l: f64,
    pub e_na: f64,
    pub e_ca: f64,
    pub e_k: f64,
    pub e_f: f64, // If reversal (~-20 mV, mixed cation)
    pub e_l: f64,
    pub dt: f64,
    pub sub_steps: usize,
    pub gain: f64,
}

impl Default for CardiacPurkinjeFibre {
    fn default() -> Self {
        Self::new()
    }
}

impl CardiacPurkinjeFibre {
    pub fn new() -> Self {
        Self {
            v: -85.0,
            m: 0.001,
            h: 0.99,
            d: 0.001,
            f: 0.99,
            x_r: 0.01,
            y: 0.05,
            c_m: 1.0,
            g_na: 15.0,  // Fast Na
            g_cal: 0.05, // L-type Ca²⁺ (small but sustains plateau)
            g_kr: 0.015, // Rapid delayed rectifier
            g_k1: 0.4,   // Inward rectifier
            g_f: 0.01,   // Funny current (pacemaker)
            g_l: 0.03,
            e_na: 40.0,
            e_ca: 65.0,
            e_k: -90.0,
            e_f: -20.0, // Mixed Na⁺/K⁺ cation
            e_l: -50.0,
            dt: 0.5,
            sub_steps: 10, // dt_sub = 0.05 ms
            gain: 1.0,
        }
    }

    #[inline]
    fn boltz(v: f64, vh: f64, k: f64) -> f64 {
        1.0 / (1.0 + (-(v - vh) / k).exp())
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let dt_sub = self.dt / self.sub_steps as f64;
        let v_prev = self.v;

        for _ in 0..self.sub_steps {
            let v = self.v;

            // Na m gate (fast)
            let m_inf = Self::boltz(v, -40.0, 8.0);
            let tau_m = 0.05 + 0.3 / (1.0 + ((v + 40.0) / 10.0).powi(2)).max(0.01);
            self.m += dt_sub * (m_inf - self.m) / tau_m;

            // Na h gate (inactivation)
            let h_inf = Self::boltz(v, -65.0, -7.0);
            let tau_h = 0.5 + 8.0 / (1.0 + ((v + 65.0) / 15.0).powi(2)).max(0.01);
            self.h += dt_sub * (h_inf - self.h) / tau_h;

            // CaL d gate (activation)
            let d_inf = Self::boltz(v, -10.0, 6.0);
            let tau_d = 2.0 + 5.0 / (1.0 + ((v + 10.0) / 10.0).powi(2)).max(0.01);
            self.d += dt_sub * (d_inf - self.d) / tau_d;

            // CaL f gate (inactivation, slow)
            let f_inf = Self::boltz(v, -30.0, -8.0);
            let tau_f = 20.0 + 100.0 / (1.0 + ((v + 30.0) / 10.0).powi(2)).max(0.01);
            self.f += dt_sub * (f_inf - self.f) / tau_f;

            // IKr x_r gate (slow activation)
            let xr_inf = Self::boltz(v, -20.0, 10.0);
            let tau_xr = 50.0 + 200.0 / (1.0 + ((v + 20.0) / 15.0).powi(2)).max(0.01);
            self.x_r += dt_sub * (xr_inf - self.x_r) / tau_xr;

            // If y gate (activates at hyperpolarised V)
            let y_inf = Self::boltz(v, -80.0, -10.0);
            let tau_y = 100.0 + 500.0 / (1.0 + ((v + 80.0) / 20.0).powi(2)).max(0.01);
            self.y += dt_sub * (y_inf - self.y) / tau_y;

            // Clamp gates
            self.m = self.m.clamp(0.0, 1.0);
            self.h = self.h.clamp(0.0, 1.0);
            self.d = self.d.clamp(0.0, 1.0);
            self.f = self.f.clamp(0.0, 1.0);
            self.x_r = self.x_r.clamp(0.0, 1.0);
            self.y = self.y.clamp(0.0, 1.0);

            // IK1: inward rectifier (voltage-dependent, Boltzmann)
            let k1_inf = 1.0 / (1.0 + ((v - self.e_k + 10.0) / 10.0).exp());

            // Currents
            let i_na = self.g_na * self.m.powi(3) * self.h * (v - self.e_na);
            let i_cal = self.g_cal * self.d * self.f * (v - self.e_ca);
            let i_kr = self.g_kr * self.x_r * (v - self.e_k);
            let i_k1 = self.g_k1 * k1_inf * (v - self.e_k);
            let i_f = self.g_f * self.y * (v - self.e_f);
            let i_l = self.g_l * (v - self.e_l);

            let dv = (-(i_na + i_cal + i_kr + i_k1 + i_f + i_l) + input) / self.c_m;
            self.v += dt_sub * dv;
        }

        // Safety
        self.v = self.v.clamp(-120.0, 60.0);
        if !self.v.is_finite() {
            self.v = -85.0;
        }
        if !self.m.is_finite() {
            self.m = 0.001;
        }
        if !self.h.is_finite() {
            self.h = 0.99;
        }
        if !self.d.is_finite() {
            self.d = 0.001;
        }
        if !self.f.is_finite() {
            self.f = 0.99;
        }
        if !self.x_r.is_finite() {
            self.x_r = 0.01;
        }
        if !self.y.is_finite() {
            self.y = 0.05;
        }

        // Spike: V crosses -20 mV upward
        if self.v >= -20.0 && v_prev < -20.0 {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Cardiac Purkinje Fibre tests --

    #[test]
    fn cardiac_fires_with_input() {
        let mut n = CardiacPurkinjeFibre::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(5.0);
        }
        assert!(
            spikes > 0,
            "Cardiac Purkinje must fire with input, got {spikes}"
        );
    }

    #[test]
    fn cardiac_silent_without_input() {
        // Without If-driven pacemaking (test with g_f=0)
        let mut n = CardiacPurkinjeFibre::new();
        n.g_f = 0.0; // Disable pacemaker
        let mut spikes = 0;
        for _ in 0..5_000 {
            spikes += n.step(0.0);
        }
        assert!(
            spikes <= 1,
            "Must be essentially silent without pacemaker, got {spikes}"
        );
    }

    #[test]
    fn cardiac_has_funny_current() {
        // If (HCN) is the hallmark pacemaker current
        let n = CardiacPurkinjeFibre::new();
        assert!(n.g_f > 0.0, "Must have funny current (If/HCN)");
    }

    #[test]
    fn cardiac_has_cal() {
        // L-type Ca²⁺ sustains the plateau
        let n = CardiacPurkinjeFibre::new();
        assert!(n.g_cal > 0.0, "Must have L-type Ca²⁺ for plateau");
    }

    #[test]
    fn cardiac_has_inward_rectifier() {
        // IK1 stabilises resting potential
        let n = CardiacPurkinjeFibre::new();
        assert!(n.g_k1 > 0.0, "Must have IK1 inward rectifier");
    }

    #[test]
    fn cardiac_six_currents() {
        let n = CardiacPurkinjeFibre::new();
        assert!(
            n.g_na > 0.0
                && n.g_cal > 0.0
                && n.g_kr > 0.0
                && n.g_k1 > 0.0
                && n.g_f > 0.0
                && n.g_l > 0.0,
            "Must have all 6 currents"
        );
    }

    #[test]
    fn cardiac_gating_evolves() {
        let mut n = CardiacPurkinjeFibre::new();
        let d0 = n.d;
        let y0 = n.y;
        for _ in 0..200 {
            n.step(5.0);
        }
        assert!(n.d != d0 || n.y != y0, "Gating must evolve");
    }

    #[test]
    fn cardiac_nan_input_stays_finite() {
        let mut n = CardiacPurkinjeFibre::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
    }

    #[test]
    fn cardiac_reset_clears_state() {
        let mut n = CardiacPurkinjeFibre::new();
        for _ in 0..500 {
            n.step(5.0);
        }
        n.reset();
        assert_eq!(n.v, -85.0);
        assert_eq!(n.m, 0.001);
    }

    #[test]
    fn cardiac_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = CardiacPurkinjeFibre::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(3.0));
        }
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 50, "1k steps must complete in <50ms");
    }

    #[test]
    fn cardiac_default_matches_constructor() {
        let default = CardiacPurkinjeFibre::default();
        let constructed = CardiacPurkinjeFibre::new();
        assert_eq!(default.v, constructed.v);
        assert_eq!(default.g_f, constructed.g_f);
        assert_eq!(default.sub_steps, constructed.sub_steps);
    }
}
