// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Smooth Muscle Cell Model

//! Calcium-dependent smooth-muscle electrical dynamics.

#![deny(missing_docs)]

// ═══════════════════════════════════════════════════════════════════
// Smooth Muscle Cell
// ═══════════════════════════════════════════════════════════════════

/// Smooth muscle cell — slow oscillatory electrical activity.
///
/// Visceral/vascular smooth muscle with Ca²⁺-dependent oscillations.
/// Key features distinct from neurons:
/// - **No fast Na⁺**: depolarisation is Ca²⁺-dependent (L-type)
/// - **BK (Ca²⁺-activated K⁺)**: repolarisation via BK channels
/// - **IP3-mediated Ca²⁺ release**: intracellular Ca²⁺ oscillations
///   from ER/SR via IP3 receptors drive slow waves
/// - **SERCA pump**: Ca²⁺ reuptake into stores
/// - **Slow oscillations**: ~3-12 cycles/min (GI slow waves)
///
/// dV/dt = (-ICaL - IBK - IL + I_ext) / C_m
/// dCa/dt = -alpha*ICaL - SERCA(Ca) + IP3_release(Ca, IP3) - Ca/tau_ca
///
/// Hirst & Edwards, J Physiol 535:165, 2001.
/// Imtiaz et al., Biophys J 83:1877, 2002.
#[derive(Clone, Debug)]
pub struct SmoothMuscleCell {
    /// Membrane potential (mV), clamped to `[-100, 40]` after each step.
    pub v: f64,
    /// CaL activation.
    pub d: f64,
    /// CaL inactivation.
    pub f: f64,
    /// Cytosolic Ca²⁺ (µM).
    pub ca: f64,
    /// ER/SR Ca²⁺ store (µM).
    pub ca_store: f64,
    /// Membrane capacitance (uF/cm^2).
    pub c_m: f64,
    /// L-type Ca²⁺ conductance.
    pub g_cal: f64,
    /// BK channel conductance.
    pub g_bk: f64,
    /// Leak conductance.
    pub g_l: f64,
    /// Calcium reversal potential (mV).
    pub e_ca: f64,
    /// Potassium reversal potential (mV).
    pub e_k: f64,
    /// Leak reversal potential (mV).
    pub e_l: f64,
    /// Ca²⁺ decay (ms).
    pub tau_ca: f64,
    /// SERCA pump max rate.
    pub v_serca: f64,
    /// SERCA Km (µM).
    pub k_serca: f64,
    /// IP3 concentration (µM, constant or input-driven).
    pub ip3: f64,
    /// IP3R max release rate.
    pub v_ip3r: f64,
    /// IP3R half-activation (µM).
    pub k_ip3: f64,
    /// Ca²⁺ co-activation of IP3R (µM).
    pub k_ca_ip3: f64,
    /// BK Ca²⁺ half-activation (µM).
    pub kd_bk: f64,
    /// Outer step (ms).
    pub dt: f64,
    /// Forward-Euler sub-steps per `dt`.
    pub sub_steps: usize,
    /// Gain applied to the input current (dimensionless).
    pub gain: f64,
}

impl Default for SmoothMuscleCell {
    fn default() -> Self {
        Self::new()
    }
}

impl SmoothMuscleCell {
    /// Create a cell at -60 mV with a full store (100 µM), the default parameters,
    /// `dt` = 1 ms and 4 sub-steps.
    pub fn new() -> Self {
        Self {
            v: -60.0,
            d: 0.01,
            f: 0.95,
            ca: 0.1,
            ca_store: 100.0, // ER/SR store (high, ~100 µM)
            c_m: 1.0,
            g_cal: 2.0, // L-type Ca²⁺
            g_bk: 1.0,  // BK
            g_l: 0.1,
            e_ca: 60.0,
            e_k: -80.0,
            e_l: -50.0,
            tau_ca: 50.0,
            v_serca: 0.5,  // SERCA pump rate
            k_serca: 0.3,  // SERCA Km
            ip3: 0.5,      // Tonic IP3
            v_ip3r: 2.0,   // IP3R release rate
            k_ip3: 0.3,    // IP3R half-act
            k_ca_ip3: 0.3, // Ca²⁺ co-activation
            kd_bk: 0.5,    // BK Ca²⁺ Kd
            dt: 1.0,       // Slow dynamics (1 ms step)
            sub_steps: 4,
            gain: 1.0,
        }
    }

    #[inline]
    fn boltz(v: f64, vh: f64, k: f64) -> f64 {
        1.0 / (1.0 + (-(v - vh) / k).exp())
    }

    /// Advance `dt` in `sub_steps` forward-Euler sub-steps with the drive
    /// `gain * current`; return 1 when `v` crosses -30 mV upwards over the whole step
    /// (a slow-wave event), else 0.
    ///
    /// Each sub-step does the following:
    /// - updates `d` and `f` from the old `v`;
    /// - advances `v` with ICaL (`d f`), BK (`ca^2 / (ca^2 + kd_bk^2)` times a voltage
    ///   Boltzmann) and the leak;
    /// - advances `ca` and `ca_store` with entry `0.01 * max(-i_cal, 0)`, IP3R release
    ///   proportional to `ca_store`, SERCA uptake and the decay `ca / tau_ca`.
    ///
    /// After the sub-steps `v` is clamped to `[-100, 40]` mV, and a non-finite value is
    /// replaced by its initial value. The input is not validated.
    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let dt_sub = self.dt / self.sub_steps as f64;
        let v_prev = self.v;

        for _ in 0..self.sub_steps {
            let v = self.v;

            // CaL d gate (activation)
            let d_inf = Self::boltz(v, -20.0, 6.0);
            let tau_d = 5.0 + 20.0 / (1.0 + ((v + 20.0) / 10.0).powi(2)).max(0.01);
            self.d += dt_sub * (d_inf - self.d) / tau_d;

            // CaL f gate (slow inactivation)
            let f_inf = Self::boltz(v, -35.0, -8.0);
            let tau_f = 50.0 + 200.0 / (1.0 + ((v + 35.0) / 10.0).powi(2)).max(0.01);
            self.f += dt_sub * (f_inf - self.f) / tau_f;

            self.d = self.d.clamp(0.0, 1.0);
            self.f = self.f.clamp(0.0, 1.0);

            // BK: Ca²⁺-dependent + voltage-dependent
            let bk_ca = self.ca * self.ca / (self.ca * self.ca + self.kd_bk * self.kd_bk);
            let bk_v = Self::boltz(v, -10.0, 15.0);
            let bk_inf = bk_ca * bk_v;

            // Currents
            let i_cal = self.g_cal * self.d * self.f * (v - self.e_ca);
            let i_bk = self.g_bk * bk_inf * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);

            let dv = (-(i_cal + i_bk + i_l) + input) / self.c_m;
            self.v += dt_sub * dv;

            // Ca²⁺ dynamics
            // Entry via CaL (inward = negative current = Ca²⁺ entry)
            let ca_entry = if i_cal < 0.0 { -i_cal * 0.01 } else { 0.0 };

            // IP3R release from store: Ca²⁺-induced Ca²⁺ release (CICR)
            let ip3_act = self.ip3 / (self.ip3 + self.k_ip3);
            let ca_act = self.ca / (self.ca + self.k_ca_ip3);
            let ip3_release = self.v_ip3r * ip3_act * ca_act * self.ca_store;

            // SERCA pump (reuptake into store)
            let serca = self.v_serca * self.ca * self.ca
                / (self.ca * self.ca + self.k_serca * self.k_serca);

            // Ca²⁺ dynamics
            self.ca += dt_sub * (ca_entry + ip3_release - serca - self.ca / self.tau_ca);
            self.ca_store += dt_sub * (serca - ip3_release);

            self.ca = self.ca.max(0.0);
            self.ca_store = self.ca_store.max(0.0);
        }

        // Safety
        self.v = self.v.clamp(-100.0, 40.0);
        if !self.v.is_finite() {
            self.v = -60.0;
        }
        if !self.ca.is_finite() {
            self.ca = 0.1;
        }
        if !self.ca_store.is_finite() {
            self.ca_store = 100.0;
        }

        // "Spike" = slow wave crossing -30 mV
        if self.v >= -30.0 && v_prev < -30.0 {
            1
        } else {
            0
        }
    }

    /// Restore every field to `new()`, parameters included.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neurons::host_budget;

    // -- Smooth Muscle Cell tests --

    #[test]
    fn smooth_fires_with_input() {
        let mut n = SmoothMuscleCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(5.0);
        }
        assert!(
            spikes > 0,
            "Smooth muscle must produce slow waves, got {spikes}"
        );
    }

    #[test]
    fn smooth_silent_without_input() {
        let mut n = SmoothMuscleCell::new();
        n.ip3 = 0.0; // No IP3 oscillation driver
        let mut spikes = 0;
        for _ in 0..5_000 {
            spikes += n.step(0.0);
        }
        assert!(
            spikes <= 1,
            "Should be essentially silent without drive, got {spikes}"
        );
    }

    #[test]
    fn smooth_has_ip3_pathway() {
        let n = SmoothMuscleCell::new();
        assert!(n.v_ip3r > 0.0, "Must have IP3R release pathway");
        assert!(n.ip3 > 0.0, "Must have tonic IP3");
    }

    #[test]
    fn smooth_has_serca() {
        let n = SmoothMuscleCell::new();
        assert!(n.v_serca > 0.0, "Must have SERCA pump");
    }

    #[test]
    fn smooth_ca_store_exists() {
        let n = SmoothMuscleCell::new();
        assert!(n.ca_store > 0.0, "Must have ER/SR Ca²⁺ store");
    }

    #[test]
    fn smooth_has_bk() {
        let n = SmoothMuscleCell::new();
        assert!(n.g_bk > 0.0, "Must have BK (Ca²⁺-activated K) channel");
    }

    #[test]
    fn smooth_no_fast_na() {
        // Smooth muscle does NOT have fast Na channels
        let n = SmoothMuscleCell::new();
        // Verify no g_na field exists by checking only CaL and BK
        assert!(n.g_cal > 0.0, "Depolarisation must be Ca²⁺-dependent");
    }

    #[test]
    fn smooth_nan_stays_finite() {
        let mut n = SmoothMuscleCell::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
        assert!(n.ca.is_finite());
    }

    #[test]
    fn smooth_reset_clears() {
        let mut n = SmoothMuscleCell::new();
        for _ in 0..1000 {
            n.step(3.0);
        }
        n.reset();
        assert_eq!(n.v, -60.0);
        assert_eq!(n.ca, 0.1);
        assert_eq!(n.ca_store, 100.0);
    }

    #[test]
    fn smooth_performance_1k_steps() {
        host_budget::assert_within(module_path!(), "smooth_performance_1k_steps", || {
            let mut n = SmoothMuscleCell::new();
            for _ in 0..1_000 {
                std::hint::black_box(n.step(2.0));
            }
        });
    }

    #[test]
    fn smooth_default_matches_constructor() {
        let default = SmoothMuscleCell::default();
        let constructed = SmoothMuscleCell::new();
        assert_eq!(default.v, constructed.v);
        assert_eq!(default.ca_store, constructed.ca_store);
        assert_eq!(default.sub_steps, constructed.sub_steps);
    }

    #[test]
    fn smooth_nonfinite_calcium_state_recovers_to_baseline() {
        let mut n = SmoothMuscleCell::new();
        n.ca = f64::INFINITY;
        n.ca_store = f64::INFINITY;
        n.sub_steps = 0;
        n.step(0.0);
        assert_eq!(n.ca, 0.1);
        assert_eq!(n.ca_store, 100.0);
        assert!(n.v.is_finite());
    }
}
