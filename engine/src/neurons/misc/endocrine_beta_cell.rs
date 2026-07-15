// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Endocrine Beta Cell Model

//! Glucose-dependent pancreatic beta-cell bursting dynamics.

// ═══════════════════════════════════════════════════════════════════
// Endocrine Beta Cell (Pancreatic)
// ═══════════════════════════════════════════════════════════════════

/// Pancreatic beta cell — glucose-dependent bursting.
///
/// Beta cells in the islets of Langerhans secrete insulin in response
/// to elevated glucose. The electrical signature is bursting: clusters
/// of spikes on a slow wave, driven by:
/// - **ICaL** (L-type Ca²⁺): spike depolarisation
/// - **IK_dr** (delayed rectifier K): spike repolarisation
/// - **IK_ATP** (ATP-sensitive K): glucose-dependent, closes with
///   rising glucose (metabolic coupling)
/// - **IK_Ca** (Ca²⁺-activated K, SK): slow burst termination
/// - **IL** (leak)
///
/// Burst mechanism: Ca²⁺ accumulates during spike burst → SK
/// activates → hyperpolarises → Ca²⁺ decays → SK deactivates →
/// next burst. IK_ATP sets the threshold: low glucose → open
/// IK_ATP → silent; high glucose → closed IK_ATP → bursting.
///
/// Chay & Keizer, Biophys J 42:181, 1983.
/// Sherman et al., Biophys J 54:411, 1988.
#[derive(Clone, Debug)]
pub struct EndocrineBetaCell {
    pub v: f64,
    pub n: f64,  // K_dr activation
    pub ca: f64, // Intracellular Ca²⁺ (µM)
    pub c_m: f64,
    pub g_cal: f64,  // L-type Ca²⁺
    pub g_kdr: f64,  // Delayed rectifier K
    pub g_katp: f64, // ATP-sensitive K (glucose-dependent)
    pub g_kca: f64,  // Ca²⁺-activated K (SK, burst termination)
    pub g_l: f64,
    pub e_ca: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub tau_ca: f64,    // Ca²⁺ decay (ms)
    pub kd_kca: f64,    // SK Ca²⁺ Kd (µM)
    pub atp_level: f64, // ATP/ADP ratio (proxy for glucose, 0-1)
    pub dt: f64,
    pub sub_steps: usize,
    pub gain: f64,
}

impl Default for EndocrineBetaCell {
    fn default() -> Self {
        Self::new()
    }
}

impl EndocrineBetaCell {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            n: 0.01,
            ca: 0.1,
            c_m: 1.0,
            g_cal: 5.0,  // L-type Ca²⁺ (strong, no Na for depolarisation)
            g_kdr: 4.0,  // Delayed rectifier
            g_katp: 3.0, // ATP-sensitive K (max conductance)
            g_kca: 2.0,  // SK for burst termination
            g_l: 0.1,
            e_ca: 50.0,
            e_k: -75.0,
            e_l: -30.0,     // Depolarised leak (typical for beta cells)
            tau_ca: 100.0,  // Ca²⁺ decay (ms)
            kd_kca: 0.5,    // SK Kd
            atp_level: 0.3, // Moderate glucose → some IK_ATP open
            dt: 0.5,
            sub_steps: 4,
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

            // CaL (instantaneous m, m_inf Boltzmann)
            let m_cal_inf = Self::boltz(v, -20.0, 8.0);

            // K_dr n gate
            let n_inf = Self::boltz(v, -15.0, 6.0);
            let tau_n = 5.0 + 20.0 / (1.0 + ((v + 15.0) / 10.0).powi(2)).max(0.01);
            self.n += dt_sub * (n_inf - self.n) / tau_n;
            self.n = self.n.clamp(0.0, 1.0);

            // IK_ATP: closes with rising ATP (glucose)
            // g_eff = g_katp * (1 - atp_level)
            let g_katp_eff = self.g_katp * (1.0 - self.atp_level);

            // IK_Ca: Ca²⁺-dependent (Hill n=2)
            let kca_inf = self.ca * self.ca / (self.ca * self.ca + self.kd_kca * self.kd_kca);

            // Currents
            let i_cal = self.g_cal * m_cal_inf * (v - self.e_ca);
            let i_kdr = self.g_kdr * self.n.powi(4) * (v - self.e_k);
            let i_katp = g_katp_eff * (v - self.e_k);
            let i_kca = self.g_kca * kca_inf * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);

            let dv = (-(i_cal + i_kdr + i_katp + i_kca + i_l) + input) / self.c_m;
            self.v += dt_sub * dv;

            // Ca²⁺ dynamics
            let ca_entry = if i_cal < 0.0 { -i_cal * 0.002 } else { 0.0 };
            self.ca += dt_sub * (ca_entry - self.ca / self.tau_ca);
            self.ca = self.ca.max(0.0);
        }

        // Safety
        self.v = self.v.clamp(-100.0, 40.0);
        if !self.v.is_finite() {
            self.v = -70.0;
        }
        if !self.n.is_finite() {
            self.n = 0.01;
        }
        if !self.ca.is_finite() {
            self.ca = 0.1;
        }

        // Spike: V crosses -20 mV
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

    // -- Endocrine Beta Cell tests --

    #[test]
    fn beta_fires_with_glucose() {
        let mut n = EndocrineBetaCell::new();
        n.atp_level = 0.9; // High glucose → closed IK_ATP → excitable
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(5.0);
        }
        assert!(
            spikes > 0,
            "Beta cell must burst with high glucose, got {spikes}"
        );
    }

    #[test]
    fn beta_silent_low_glucose() {
        let mut n = EndocrineBetaCell::new();
        n.atp_level = 0.0; // Low glucose → open IK_ATP → silent
        let mut spikes = 0;
        for _ in 0..5_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(
            spikes, 0,
            "Beta cell must be silent at low glucose, got {spikes}"
        );
    }

    #[test]
    fn beta_katp_glucose_dependent() {
        // Higher glucose (more ATP) → less IK_ATP → more excitable
        let mut low = EndocrineBetaCell::new();
        low.atp_level = 0.2;
        let mut high = EndocrineBetaCell::new();
        high.atp_level = 0.9;
        let (mut sl, mut sh) = (0, 0);
        for _ in 0..5_000 {
            sl += low.step(1.0);
            sh += high.step(1.0);
        }
        assert!(
            sh >= sl,
            "High glucose → more spikes: high={sh} vs low={sl}"
        );
    }

    #[test]
    fn beta_has_katp() {
        let n = EndocrineBetaCell::new();
        assert!(n.g_katp > 0.0, "Must have ATP-sensitive K channel");
    }

    #[test]
    fn beta_has_kca_for_bursting() {
        let n = EndocrineBetaCell::new();
        assert!(n.g_kca > 0.0, "Must have IK_Ca for burst termination");
    }

    #[test]
    fn beta_ca_rises_with_spiking() {
        let mut n = EndocrineBetaCell::new();
        n.atp_level = 0.8;
        let ca0 = n.ca;
        for _ in 0..5_000 {
            n.step(2.0);
        }
        assert!(
            n.ca > ca0,
            "Ca²⁺ should rise during bursting: ca0={ca0}, ca={}",
            n.ca
        );
    }

    #[test]
    fn beta_nan_stays_finite() {
        let mut n = EndocrineBetaCell::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
        assert!(n.ca.is_finite());
    }

    #[test]
    fn beta_reset_clears() {
        let mut n = EndocrineBetaCell::new();
        for _ in 0..1000 {
            n.step(2.0);
        }
        n.reset();
        assert_eq!(n.v, -70.0);
        assert_eq!(n.ca, 0.1);
    }

    #[test]
    fn beta_no_fast_na() {
        // Beta cells do not have fast Na — depolarisation is CaL-dependent
        let n = EndocrineBetaCell::new();
        assert!(n.g_cal > 0.0, "CaL must drive depolarisation");
    }

    #[test]
    fn beta_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = EndocrineBetaCell::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(2.0));
        }
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 50, "1k steps must complete in <50ms");
    }

    #[test]
    fn beta_default_matches_constructor() {
        let default = EndocrineBetaCell::default();
        let constructed = EndocrineBetaCell::new();
        assert_eq!(default.v, constructed.v);
        assert_eq!(default.atp_level, constructed.atp_level);
        assert_eq!(default.sub_steps, constructed.sub_steps);
    }

    #[test]
    fn beta_nonfinite_calcium_recovers_to_baseline() {
        let mut n = EndocrineBetaCell::new();
        n.ca = f64::INFINITY;
        n.sub_steps = 0;
        n.step(0.0);
        assert_eq!(n.ca, 0.1);
        assert!(n.v.is_finite());
    }
}
