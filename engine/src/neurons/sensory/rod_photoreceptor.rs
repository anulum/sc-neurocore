// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sensory Neuron Models

// ═══════════════════════════════════════════════════════════════════
// Rod Photoreceptor — scotopic vision
// ═══════════════════════════════════════════════════════════════════

/// Rod photoreceptor — scotopic vision with Ca²⁺ feedback.
///
/// Phototransduction cascade per Nikonov et al. 2006:
/// 1. Light → rhodopsin → transducin → PDE activation
/// 2. PDE hydrolyses cGMP → CNG channels close → hyperpolarise
/// 3. Ca²⁺ enters via CNG channels (dark current)
/// 4. Ca²⁺ feedback on guanylyl cyclase (GC): low Ca²⁺ → more cGMP
///    production → light adaptation (Ca²⁺-feedback is the key
///    mechanism for rod sensitivity regulation)
///
/// dcGMP/dt = alpha_GC(Ca) - beta_PDE(light)*cGMP
/// dCa/dt = eta*J_CNG(cGMP) - Ca/tau_Ca
///
/// where alpha_GC(Ca) = alpha_max * K_gc^n / (K_gc^n + Ca^n)
/// is the Ca²⁺-dependent GC activity (Hill inhibition).
///
/// Graded, no spikes. Very slow recovery.
///
/// Nikonov et al., J Gen Physiol 127:359, 2006.
/// Hamer et al., J Gen Physiol 125:287, 2005.
#[derive(Clone, Debug)]
pub struct RodPhotoreceptor {
    pub v: f64,
    pub v_dark: f64,
    pub v_hyper: f64,
    pub cgmp: f64,    // cGMP concentration (normalised)
    pub ca: f64,      // Ca²⁺ concentration (normalised, ~1.0 in dark)
    pub tau_act: f64, // PDE activation time constant (ms)
    pub tau_ca: f64,  // Ca²⁺ extrusion time constant (ms)
    pub sensitivity: f64,
    pub alpha_max: f64, // Max GC synthesis rate
    pub k_gc: f64,      // Ca²⁺ half-inhibition of GC
    pub n_gc: f64,      // Hill coefficient for GC inhibition
    pub eta_ca: f64,    // Ca²⁺ entry per unit CNG current
    pub dt: f64,
}

impl RodPhotoreceptor {
    pub fn new() -> Self {
        Self {
            v: -40.0,
            v_dark: -40.0,
            v_hyper: -70.0,
            cgmp: 1.0,
            ca: 1.0, // High Ca²⁺ in dark (CNG channels open)
            tau_act: 20.0,
            tau_ca: 30.0, // Ca²⁺ extrusion (~30 ms, NCKX exchanger)
            sensitivity: 0.01,
            alpha_max: 0.05, // Max cGMP synthesis rate
            k_gc: 0.5,       // Ca²⁺ half-inhibition of GC
            n_gc: 4.0,       // Hill coefficient (cooperative, Nikonov 2006)
            eta_ca: 0.3,     // Ca²⁺ entry gain
            dt: 0.1,
        }
    }

    /// Ca²⁺-dependent guanylyl cyclase rate (Hill inhibition).
    /// Low Ca²⁺ → high GC activity → more cGMP → adaptation.
    #[inline]
    fn gc_rate(&self) -> f64 {
        let ca_n = self.ca.powf(self.n_gc);
        let k_n = self.k_gc.powf(self.n_gc);
        self.alpha_max * k_n / (k_n + ca_n)
    }

    /// Step with light intensity (≥ 0). Returns membrane potential (mV).
    pub fn step(&mut self, light: f64) -> f64 {
        let light_clamped = light.max(0.0);

        // cGMP dynamics: synthesis (GC, Ca²⁺-dependent) - hydrolysis (PDE, light-driven)
        let gc = self.gc_rate();
        let pde = self.sensitivity * light_clamped / self.tau_act;
        let d_cgmp = gc - pde * self.cgmp + (1.0 - self.cgmp) * 0.001; // Basal turnover
        self.cgmp += d_cgmp * self.dt;
        self.cgmp = self.cgmp.clamp(0.0, 1.5); // Can transiently overshoot during adaptation

        // CNG current proportional to cGMP^3
        let cng_fraction = self.cgmp.powi(3).min(1.0);

        // Ca²⁺ dynamics: entry via CNG - extrusion via NCKX
        let d_ca = self.eta_ca * cng_fraction - self.ca / self.tau_ca;
        self.ca += d_ca * self.dt;
        self.ca = self.ca.max(0.0);

        // Membrane potential
        self.v = self.v_hyper + (self.v_dark - self.v_hyper) * cng_fraction;
        if !self.v.is_finite() {
            self.v = self.v_dark;
        }
        if !self.cgmp.is_finite() {
            self.cgmp = 1.0;
        }
        if !self.ca.is_finite() {
            self.ca = 1.0;
        }
        self.v
    }

    pub fn reset(&mut self) {
        self.v = self.v_dark;
        self.cgmp = 1.0;
        self.ca = 1.0;
    }
}

impl Default for RodPhotoreceptor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rod_hyperpolarises_with_light() {
        let mut r = RodPhotoreceptor::new();
        let v_dark = r.v;
        for _ in 0..1000 {
            r.step(100.0);
        }
        assert!(r.v < v_dark, "rod should hyperpolarise: v={}", r.v);
    }

    #[test]
    fn rod_stays_dark_without_light() {
        let mut r = RodPhotoreceptor::new();
        for _ in 0..500 {
            r.step(0.0);
        }
        assert!((r.v - r.v_dark).abs() < 1.0);
    }

    #[test]
    fn rod_slow_recovery() {
        let mut r = RodPhotoreceptor::new();
        // Flash
        for _ in 0..500 {
            r.step(200.0);
        }
        let v_after_flash = r.v;
        // Dark: slow recovery
        for _ in 0..1000 {
            r.step(0.0);
        }
        assert!(r.v > v_after_flash, "rod should recover in dark");
        assert!(r.v < r.v_dark, "rod should not fully recover in 1000 steps");
    }

    #[test]
    fn rod_cgmp_bounded() {
        let mut r = RodPhotoreceptor::new();
        for _ in 0..10000 {
            r.step(1000.0);
        }
        assert!(
            r.cgmp >= 0.0 && r.cgmp <= 1.5,
            "cGMP should be bounded: {}",
            r.cgmp
        );
        r.reset();
        for _ in 0..10000 {
            r.step(-10.0);
        } // Negative light clamped to 0
          // With Ca²⁺ feedback, cGMP can transiently overshoot during adaptation
        assert!(
            r.cgmp >= 0.0 && r.cgmp <= 1.5,
            "cGMP should be bounded: {}",
            r.cgmp
        );
    }

    #[test]
    fn rod_ca_feedback_adaptation() {
        // Ca²⁺ feedback should cause light adaptation:
        // sustained light → Ca²⁺ drops → GC increases → cGMP partially recovers
        let mut r = RodPhotoreceptor::new();
        // Apply light
        for _ in 0..5000 {
            r.step(100.0);
        }
        let v_adapted = r.v;
        let ca_adapted = r.ca;
        // Ca²⁺ should be lower than dark level
        assert!(
            ca_adapted < 1.0,
            "Ca²⁺ should drop during light: ca={ca_adapted:.3}"
        );
        // V should not be fully hyperpolarised (adaptation compensates)
        assert!(
            v_adapted > r.v_hyper + 1.0,
            "Adaptation should partially restore: v={v_adapted:.1}, v_hyper={}",
            r.v_hyper
        );
    }

    #[test]
    fn rod_performance() {
        let mut r = RodPhotoreceptor::new();
        let start = std::time::Instant::now();
        for _ in 0..100_000 {
            r.step(50.0);
        }
        assert!(start.elapsed().as_millis() < 50);
    }

    #[test]
    fn rod_photoreceptor_default_matches_constructor_contract() {
        let default = RodPhotoreceptor::default();
        let constructed = RodPhotoreceptor::new();
        assert_eq!(default.v, constructed.v);
        assert_eq!(default.cgmp, constructed.cgmp);
        assert_eq!(default.ca, constructed.ca);
        assert_eq!(default.dt, constructed.dt);
    }

    #[test]
    fn rod_nonfinite_runtime_recovers_safe_defaults() {
        let mut voltage = RodPhotoreceptor::new();
        voltage.v_hyper = f64::NAN;
        assert_eq!(voltage.step(0.0), voltage.v_dark);

        let mut cgmp = RodPhotoreceptor::new();
        cgmp.ca = f64::NAN;
        cgmp.step(0.0);
        assert_eq!(cgmp.cgmp, 1.0);
        assert_eq!(cgmp.ca, 0.0);

        let mut calcium = RodPhotoreceptor::new();
        calcium.eta_ca = f64::INFINITY;
        calcium.step(0.0);
        assert_eq!(calcium.ca, 1.0);
    }
}
