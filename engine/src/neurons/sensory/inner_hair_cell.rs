// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sensory Neuron Models

// ═══════════════════════════════════════════════════════════════════
// Inner Hair Cell (IHC) — auditory
// ═══════════════════════════════════════════════════════════════════

/// Inner hair cell — primary auditory transducer.
///
/// Mechano-electrical transduction: stereocilia displacement opens
/// MET channels → depolarisation → Ca2+ influx → glutamate release.
/// Inner hair cell with Meddis vesicle pool dynamics.
///
/// Three-stage model per Meddis (1986, 2006):
/// 1. **MET transduction**: Boltzmann gating of mechano-electrical
///    transducer channels converts stereocilia displacement to
///    receptor potential.
/// 2. **Ca²⁺ dynamics**: voltage-dependent Ca²⁺ entry drives
///    vesicle release.
/// 3. **Meddis vesicle pool**: three-compartment transmitter model:
///    - q: available vesicles (free pool)
///    - c: cleft transmitter concentration
///    - w: reprocessing store (depleted vesicles recovering)
///
///    dq/dt = y*(M-q) + x_r*w - k*q*f(Ca)    (replenishment + recovery - release)
///    dc/dt = k*q*f(Ca) - l*c - r_up*c        (release - loss - reuptake)
///    dw/dt = r_up*c - x_r*w                   (reuptake - recovery)
///
///    where f(Ca) = Ca²/(Ca² + K_d²) is Ca²⁺-dependent release rate.
///
/// Graded output: receptor potential (no spikes).
///
/// Meddis, JASA 79:702, 1986.
/// Meddis, JASA 119:406, 2006.
/// Lopez-Poveda & Eustaquio-Martín, JASA 119:416, 2006.
#[derive(Clone, Debug)]
pub struct InnerHairCell {
    // Membrane
    pub v: f64, // Receptor potential (mV)
    pub v_rest: f64,
    pub tau: f64,    // Membrane time constant (ms)
    pub g_met: f64,  // MET channel max conductance
    pub x_half: f64, // Boltzmann half-activation (nm)
    pub s_met: f64,  // Boltzmann slope
    // Ca²⁺
    pub ca: f64,        // Intracellular Ca²⁺ (µM)
    pub tau_ca: f64,    // Ca²⁺ decay time constant (ms)
    pub g_ca: f64,      // Ca²⁺ entry gain
    pub v_ca_half: f64, // Ca²⁺ channel half-activation (mV)
    pub s_ca: f64,      // Ca²⁺ channel slope
    // Meddis vesicle pool
    pub q: f64,      // Available vesicles (free pool) [0, M]
    pub c: f64,      // Cleft transmitter concentration
    pub w: f64,      // Reprocessing store
    pub m_pool: f64, // Maximum vesicle pool size
    pub y: f64,      // Replenishment rate (ms⁻¹)
    pub x_r: f64,    // Recovery rate from reprocessing (ms⁻¹)
    pub k_rel: f64,  // Release rate constant (ms⁻¹)
    pub l: f64,      // Loss rate from cleft (ms⁻¹)
    pub r_up: f64,   // Reuptake rate (ms⁻¹)
    pub k_d: f64,    // Ca²⁺ half-saturation for release (µM)
    pub dt: f64,
}

impl InnerHairCell {
    pub fn new() -> Self {
        Self {
            v: -60.0,
            v_rest: -60.0,
            tau: 0.5,
            g_met: 10.0,
            x_half: 50.0,
            s_met: 10.0,
            ca: 0.05,
            tau_ca: 1.0,
            g_ca: 0.5,
            v_ca_half: -35.0, // CaV1.3 half-activation
            s_ca: 8.0,
            // Meddis pool defaults (Meddis 2006 Table I range)
            q: 8.0,
            c: 0.0,
            w: 0.0,
            m_pool: 10.0, // Max vesicle pool
            y: 0.01,      // Replenishment (slow, ms⁻¹)
            x_r: 0.005,   // Recovery from reprocessing (ms⁻¹)
            k_rel: 0.2,   // Release rate constant (ms⁻¹)
            l: 0.05,      // Cleft loss (ms⁻¹)
            r_up: 0.05,   // Reuptake (ms⁻¹)
            k_d: 0.1,     // Ca²⁺ Kd for release (µM)
            dt: 0.025,
        }
    }

    /// Ca²⁺-dependent release function: Hill (n=2).
    #[inline]
    fn release_rate(&self) -> f64 {
        let ca2 = self.ca * self.ca;
        let kd2 = self.k_d * self.k_d;
        self.k_rel * ca2 / (ca2 + kd2)
    }

    /// Step with stereocilia displacement (nm). Returns receptor potential (mV).
    pub fn step(&mut self, displacement: f64) -> f64 {
        // 1. MET transduction
        let p_open = 1.0 / (1.0 + (-(displacement - self.x_half) / self.s_met).exp());
        let i_met = self.g_met * p_open * (0.0 - self.v);
        self.v += (-(self.v - self.v_rest) + i_met) / self.tau * self.dt;

        // 2. Ca²⁺ dynamics (voltage-gated CaV1.3)
        let m_ca = 1.0 / (1.0 + (-(self.v - self.v_ca_half) / self.s_ca).exp());
        let ca_entry = self.g_ca * m_ca * m_ca; // m² activation
        self.ca += (-self.ca / self.tau_ca + ca_entry) * self.dt;
        self.ca = self.ca.max(0.0);

        // 3. Meddis vesicle pool dynamics
        let f_ca = self.release_rate();
        let dq = self.y * (self.m_pool - self.q) + self.x_r * self.w - f_ca * self.q;
        let dc = f_ca * self.q - self.l * self.c - self.r_up * self.c;
        let dw = self.r_up * self.c - self.x_r * self.w;

        self.q += dq * self.dt;
        self.c += dc * self.dt;
        self.w += dw * self.dt;

        // Bounds
        self.q = self.q.clamp(0.0, self.m_pool);
        self.c = self.c.max(0.0);
        self.w = self.w.max(0.0);
        if !self.v.is_finite() {
            self.v = self.v_rest;
        }
        if !self.ca.is_finite() {
            self.ca = 0.05;
        }
        if !self.q.is_finite() {
            self.q = 8.0;
        }
        if !self.c.is_finite() {
            self.c = 0.0;
        }
        if !self.w.is_finite() {
            self.w = 0.0;
        }

        self.v
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.ca = 0.05;
        self.q = 8.0;
        self.c = 0.0;
        self.w = 0.0;
    }
}

impl Default for InnerHairCell {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ihc_depolarises_with_displacement() {
        let mut c = InnerHairCell::new();
        let v_rest = c.v;
        for _ in 0..200 {
            c.step(50.0);
        }
        assert!(c.v > v_rest, "IHC should depolarise: v={}", c.v);
    }

    #[test]
    fn ihc_no_change_at_zero() {
        let mut c = InnerHairCell::new();
        for _ in 0..200 {
            c.step(0.0);
        }
        assert!(
            (c.v - c.v_rest).abs() < 5.0,
            "IHC should stay near rest with no displacement"
        );
    }

    #[test]
    fn ihc_ca_increases_with_depolarisation() {
        let mut c = InnerHairCell::new();
        for _ in 0..200 {
            c.step(60.0);
        }
        assert!(c.ca > 0.0, "Ca2+ should increase during depolarisation");
    }

    #[test]
    fn ihc_reset_roundtrip() {
        let mut c = InnerHairCell::new();
        for _ in 0..100 {
            c.step(50.0);
        }
        c.reset();
        assert_eq!(c.v, c.v_rest);
        assert_eq!(c.ca, 0.05);
        assert_eq!(c.q, 8.0);
        assert_eq!(c.c, 0.0);
        assert_eq!(c.w, 0.0);
    }

    #[test]
    fn ihc_bounded() {
        let mut c = InnerHairCell::new();
        for _ in 0..10000 {
            c.step(100.0);
        }
        assert!(c.v.is_finite());
        assert!(c.ca.is_finite());
    }

    #[test]
    fn ihc_vesicle_pool_depletes() {
        // Sustained stimulation should deplete available vesicles (q)
        let mut c = InnerHairCell::new();
        let q0 = c.q;
        for _ in 0..5000 {
            c.step(80.0);
        }
        assert!(
            c.q < q0,
            "Vesicle pool should deplete: q0={q0}, q_now={}",
            c.q
        );
    }

    #[test]
    fn ihc_cleft_transmitter_rises() {
        // Stimulation should release transmitter into cleft
        let mut c = InnerHairCell::new();
        for _ in 0..2000 {
            c.step(80.0);
        }
        assert!(
            c.c > 0.0,
            "Cleft transmitter should rise with stimulation: c={}",
            c.c
        );
    }

    #[test]
    fn ihc_reprocessing_store_fills() {
        // Reuptake from cleft should fill reprocessing store
        let mut c = InnerHairCell::new();
        for _ in 0..5000 {
            c.step(80.0);
        }
        assert!(
            c.w > 0.0,
            "Reprocessing store should fill via reuptake: w={}",
            c.w
        );
    }

    #[test]
    fn ihc_pool_mass_conserved() {
        // Total transmitter (q + c + w) should not exceed m_pool
        let mut c = InnerHairCell::new();
        for _ in 0..10000 {
            c.step(80.0);
        }
        let total = c.q + c.c + c.w;
        assert!(
            total <= c.m_pool * 1.5,
            "Total transmitter should be bounded: q+c+w={total:.2}, m={}",
            c.m_pool
        );
    }

    #[test]
    fn ihc_performance() {
        let mut c = InnerHairCell::new();
        let start = std::time::Instant::now();
        for _ in 0..100_000 {
            c.step(50.0);
        }
        assert!(start.elapsed().as_millis() < 50);
    }

    #[test]
    fn inner_hair_cell_default_matches_constructor_contract() {
        let default = InnerHairCell::default();
        let constructed = InnerHairCell::new();
        assert_eq!(default.v, constructed.v);
        assert_eq!(default.ca, constructed.ca);
        assert_eq!(default.q, constructed.q);
        assert_eq!(default.dt, constructed.dt);
    }

    #[test]
    fn ihc_nonfinite_runtime_recovers_safe_defaults() {
        let mut voltage = InnerHairCell::new();
        voltage.v = f64::NAN;
        assert_eq!(voltage.step(50.0), voltage.v_rest);

        let mut calcium = InnerHairCell::new();
        calcium.g_ca = f64::INFINITY;
        calcium.step(50.0);
        assert_eq!(calcium.ca, 0.05);

        let mut available_pool = InnerHairCell::new();
        available_pool.k_rel = f64::NAN;
        available_pool.step(50.0);
        assert_eq!(available_pool.q, 8.0);

        let mut cleft = InnerHairCell::new();
        cleft.k_rel = f64::INFINITY;
        cleft.step(50.0);
        assert_eq!(cleft.c, 0.0);

        let mut reprocessing_store = InnerHairCell::new();
        reprocessing_store.c = 1.0;
        reprocessing_store.r_up = f64::INFINITY;
        reprocessing_store.step(50.0);
        assert_eq!(reprocessing_store.w, 0.0);
    }
}
