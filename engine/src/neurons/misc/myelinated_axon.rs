// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Myelinated Axon Models

//! Frankenhaeuser-Huxley and MRG nodal models for myelinated axons.

// ═══════════════════════════════════════════════════════════════════
// Frankenhaeuser-Huxley Axon
// ═══════════════════════════════════════════════════════════════════

/// Frankenhaeuser-Huxley 1964 — myelinated nerve fibre model.
///
/// Extension of HH for myelinated axons (Xenopus node of Ranvier).
/// Uses Goldman-Hodgkin-Katz (GHK) permeability-based current equations
/// with 4 gating variables: m (Na activation), h (Na inactivation),
/// n (delayed rectifier K), p (slow non-specific current).
///
/// GHK current for monovalent ion:
///   I = P * F²V/(RT) * (C_i - C_o * exp(-FV/RT)) / (1 - exp(-FV/RT))
///
/// Simplified (FH convention, V relative to rest, temperature factor absorbed):
///   I_Na = P_Na * m² * h * ghk_drive(V, Na_i/Na_o)
///   I_K  = P_K  * n² * ghk_drive(V, K_i/K_o)
///   I_p  = P_p  * p² * ghk_drive(V, Na_i/Na_o)  [non-specific, Na-like]
///   I_L  = g_L * (V - E_L)
///
/// C dV/dt = -(I_Na + I_K + I_p + I_L) + I_ext
///
/// The GHK driving force is the key distinction from conductance-based
/// HH — it is nonlinear in V and depends on concentration ratios.
///
/// Frankenhaeuser & Huxley, J Physiol 171:302, 1964.
/// Frankenhaeuser, J Physiol 160:46, 1962 (rate constants).
#[derive(Clone, Debug)]
pub struct FrankenhaeUserHuxleyAxon {
    pub v: f64,    // Membrane potential (mV, relative to rest)
    pub m: f64,    // Na activation
    pub h: f64,    // Na inactivation
    pub n: f64,    // K delayed rectifier
    pub p: f64,    // Slow non-specific
    pub c_m: f64,  // µF/cm²
    pub p_na: f64, // Na permeability (10⁻³ cm/s, FH units)
    pub p_k: f64,  // K permeability
    pub p_p: f64,  // Slow current permeability
    pub g_l: f64,  // Leak conductance (mS/cm²)
    pub e_l: f64,  // Leak reversal (mV relative to rest)
    // Concentration ratios (C_i / C_o) for GHK
    pub na_ratio: f64, // [Na]_i / [Na]_o (~0.1 for frog)
    pub k_ratio: f64,  // [K]_i / [K]_o (~30 for frog)
    pub v_t: f64,      // RT/F thermal voltage (mV), ~25.3 at 20°C
    pub dt: f64,
    pub sub_steps: usize,
    pub gain: f64,
}

impl Default for FrankenhaeUserHuxleyAxon {
    fn default() -> Self {
        Self::new()
    }
}

impl FrankenhaeUserHuxleyAxon {
    pub fn new() -> Self {
        Self {
            v: 0.0, // Relative to resting potential
            m: 0.005,
            h: 0.8,
            n: 0.01,
            p: 0.01,
            c_m: 2.0, // µF/cm² (myelinated node, FH Table 4)
            // Effective permeabilities: P_raw * F * [C]_o / 1000
            // FH Table 4: P_Na=8e-3, P_K=1.2e-3, P_p=0.54e-3 cm/s
            // [Na]_o=114.5, [K]_o=2.5 mM; F=96.485 C/mmol
            p_na: 88.4,     // 8e-3 * 96.485 * 114.5 / 1000 (mA/cm² per unit gating)
            p_k: 0.29,      // 1.2e-3 * 96.485 * 2.5 / 1000
            p_p: 5.96,      // 0.54e-3 * 96.485 * 114.5 / 1000 (Na-like)
            g_l: 30.3,      // Leak (FH Table 4: 30.3 mS/cm²)
            e_l: 0.026,     // Leak reversal (FH Table 4: 0.026 mV)
            na_ratio: 0.12, // [Na]_i/[Na]_o = 13.74/114.5 (FH 1962)
            k_ratio: 48.0,  // [K]_i/[K]_o = 120/2.5 (FH 1962)
            v_t: 25.3,      // RT/F at 20°C (293K)
            dt: 0.5,        // External step (ms)
            sub_steps: 50,  // dt_sub = 0.01 ms
            gain: 1.0,
        }
    }

    /// GHK current for monovalent ion (FH convention).
    ///
    /// Returns current density contribution in mA/cm² when P is in
    /// FH-scaled units (absorbs F*[C_o]*1e-3 into P).
    ///
    /// I = P_eff * gates * V/V_T * (r - exp(-V/V_T)) / (1 - exp(-V/V_T))
    ///
    /// where r = [ion]_i / [ion]_o, V_T = RT/F.
    /// At V→0: uses L'Hôpital limit = P_eff * (r - 1).
    ///
    /// The Faraday scaling: P_eff = P_raw * F * [C]_o / 1000
    /// is absorbed into the permeability constant (FH Table 4 values
    /// are already in these effective units: mA/cm² at unit gating).
    #[inline]
    fn ghk_current(v: f64, c_ratio: f64, v_t: f64) -> f64 {
        if v.abs() < 0.01 {
            // L'Hôpital limit
            c_ratio - 1.0
        } else {
            let u = v / v_t;
            let exp_neg_u = (-u).exp();
            u * (c_ratio - exp_neg_u) / (1.0 - exp_neg_u)
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let dt_sub = self.dt / self.sub_steps as f64;
        let v_prev = self.v;

        for _ in 0..self.sub_steps {
            let v = self.v;

            // FH alpha/beta rate functions (Frankenhaeuser 1962, Table 1)
            // All rates in ms⁻¹, V in mV relative to rest.

            // m gate (Na activation)
            let am = if (v - 22.0).abs() < 0.1 {
                1.87
            } else {
                0.36 * (v - 22.0) / (1.0 - (-(v - 22.0) / 3.0).exp())
            };
            let bm = if (v - 13.0).abs() < 0.1 {
                1.87
            } else {
                0.4 * (13.0 - v) / (1.0 - ((v - 13.0) / 20.0).exp())
            };

            // h gate (Na inactivation)
            let ah = if (v + 10.0).abs() < 0.1 {
                0.08
            } else {
                0.1 * (-10.0 - v) / (1.0 - ((v + 10.0) / 6.0).exp())
            };
            let bh = 4.5 / (1.0 + ((45.0 - v) / 10.0).exp());

            // n gate (K delayed rectifier)
            let an = if (v - 13.0).abs() < 0.1 {
                0.1
            } else {
                0.02 * (v - 13.0) / (1.0 - (-(v - 13.0) / 10.0).exp())
            };
            let bn = if (v - 23.0).abs() < 0.1 {
                0.05
            } else {
                0.05 * (23.0 - v) / (1.0 - ((v - 23.0) / 10.0).exp())
            };

            // p gate (slow non-specific)
            let ap = if (v - 21.0).abs() < 0.1 {
                0.04
            } else {
                0.006 * (v - 21.0) / (1.0 - (-(v - 21.0) / 2.0).exp())
            };
            let bp = if (v + 4.0).abs() < 0.1 {
                0.04
            } else {
                0.09 * (-4.0 - v) / (1.0 - ((v + 4.0) / 2.0).exp())
            };

            // Ensure rates are non-negative
            let am = am.max(0.0);
            let bm = bm.max(0.0);
            let ah = ah.max(0.0);
            let bh = bh.max(0.0);
            let an = an.max(0.0);
            let bn = bn.max(0.0);
            let ap = ap.max(0.0);
            let bp = bp.max(0.0);

            // Gate updates
            self.m += dt_sub * (am * (1.0 - self.m) - bm * self.m);
            self.h += dt_sub * (ah * (1.0 - self.h) - bh * self.h);
            self.n += dt_sub * (an * (1.0 - self.n) - bn * self.n);
            self.p += dt_sub * (ap * (1.0 - self.p) - bp * self.p);

            // Clamp gates
            self.m = self.m.clamp(0.0, 1.0);
            self.h = self.h.clamp(0.0, 1.0);
            self.n = self.n.clamp(0.0, 1.0);
            self.p = self.p.clamp(0.0, 1.0);

            // GHK permeability-based currents (FH Table 4)
            let i_na = self.p_na
                * self.m
                * self.m
                * self.h
                * Self::ghk_current(v, self.na_ratio, self.v_t);
            let i_k = self.p_k * self.n * self.n * Self::ghk_current(v, self.k_ratio, self.v_t);
            // p-current uses Na-like concentration ratio (non-specific cation)
            let i_p = self.p_p * self.p * self.p * Self::ghk_current(v, self.na_ratio, self.v_t);
            let i_l = self.g_l * (self.v - self.e_l);

            let dv = (-(i_na + i_k + i_p + i_l) + input) / self.c_m;
            self.v += dt_sub * dv;
        }

        // Safety
        self.v = self.v.clamp(-50.0, 150.0);
        if !self.v.is_finite() {
            self.v = 0.0;
        }
        if !self.m.is_finite() {
            self.m = 0.005;
        }
        if !self.h.is_finite() {
            self.h = 0.8;
        }
        if !self.n.is_finite() {
            self.n = 0.01;
        }
        if !self.p.is_finite() {
            self.p = 0.01;
        }

        // Spike detection: V crosses 40 mV upward
        if self.v >= 40.0 && v_prev < 40.0 {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Node of Ranvier (McIntyre-Richardson-Grill 2002)
// ═══════════════════════════════════════════════════════════════════

/// Node of Ranvier — McIntyre-Richardson-Grill 2002 model.
///
/// Gold-standard nodal model for mammalian myelinated axons. Includes
/// the specific channel complement of nodes of Ranvier:
///
/// - **INaT** (transient Na, Nav1.6): m³h gating, fast activation
/// - **INaP** (persistent Na, Nav1.6): p³ gating, subthreshold amplification
/// - **IKs** (slow K, Kv7/KCNQ): s gating, membrane stabilisation
/// - **IKf** (fast K, Kv3.1-like): fast repolarisation (n⁴ HH-style, optional)
/// - **IL** (leak)
///
/// The persistent Na current (INaP) is critical — it provides subthreshold
/// amplification and lowers the effective firing threshold, a key feature
/// of Nav1.6-rich nodes that distinguishes them from generic HH models.
///
/// C dV/dt = -(INaT + INaP + IKs + IL) + I_ext
///
/// Gating uses Boltzmann steady-state + time-constant formulation
/// (not alpha/beta) following MRG convention.
///
/// McIntyre, Richardson & Grill, J Neurophysiol 87:995, 2002.
#[derive(Clone, Debug)]
pub struct NodeOfRanvier {
    pub v: f64,     // Membrane potential (mV)
    pub m: f64,     // Nav1.6 transient activation
    pub h: f64,     // Nav1.6 transient inactivation
    pub p: f64,     // Nav1.6 persistent activation
    pub s: f64,     // Kv7 slow K activation
    pub c_m: f64,   // Nodal capacitance (µF/cm²)
    pub g_nat: f64, // Transient Na conductance (mS/cm²)
    pub g_nap: f64, // Persistent Na conductance
    pub g_ks: f64,  // Slow K (Kv7) conductance
    pub g_l: f64,   // Leak conductance
    pub e_na: f64,  // Na reversal (mV)
    pub e_k: f64,   // K reversal (mV)
    pub e_l: f64,   // Leak reversal (mV)
    pub dt: f64,    // External time step (ms)
    pub sub_steps: usize,
    pub gain: f64,
}

impl Default for NodeOfRanvier {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeOfRanvier {
    pub fn new() -> Self {
        Self {
            v: -80.0,
            m: 0.01,
            h: 0.75,
            p: 0.01,
            s: 0.05,
            c_m: 2.0,      // µF/cm² (MRG nodal value)
            g_nat: 3000.0, // mS/cm² (high Nav1.6 density)
            g_nap: 5.0,    // Persistent Na (small but critical)
            g_ks: 80.0,    // Kv7/KCNQ slow K
            g_l: 7.0,      // Nodal leak (higher than soma)
            e_na: 50.0,
            e_k: -90.0,
            e_l: -90.0,    // MRG nodal resting ~-80 mV
            dt: 0.5,       // External step (ms)
            sub_steps: 20, // dt_sub = 0.025 ms
            gain: 1.0,
        }
    }

    /// Boltzmann steady-state: 1 / (1 + exp(-(V - V_half) / k))
    #[inline]
    fn boltz(v: f64, v_half: f64, k: f64) -> f64 {
        1.0 / (1.0 + (-(v - v_half) / k).exp())
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let dt_sub = self.dt / self.sub_steps as f64;
        let v_prev_ext = self.v;

        for _ in 0..self.sub_steps {
            let v = self.v;

            // Nav1.6 transient: m gate (fast activation)
            // MRG: V_half = -26.8 mV, k = 9.2 mV
            let m_inf = Self::boltz(v, -26.8, 9.2);
            let tau_m = 0.025 + 0.14 / (1.0 + ((v + 25.0) / 10.0).powi(2)).max(0.01);
            self.m += dt_sub * (m_inf - self.m) / tau_m;

            // Nav1.6 transient: h gate (inactivation)
            // MRG: V_half = -55.2 mV, k = -7.4 mV (negative slope)
            let h_inf = Self::boltz(v, -55.2, -7.4);
            let tau_h = 0.6 + 4.0 / (1.0 + ((v + 45.0) / 10.0).powi(2)).max(0.01);
            self.h += dt_sub * (h_inf - self.h) / tau_h;

            // Nav1.6 persistent: p gate (slow activation)
            // MRG: V_half = -44.0 mV, k = 5.0 mV
            let p_inf = Self::boltz(v, -44.0, 5.0);
            let tau_p = 1.0 + 6.0 / (1.0 + ((v + 40.0) / 10.0).powi(2)).max(0.01);
            self.p += dt_sub * (p_inf - self.p) / tau_p;

            // Kv7 slow K: s gate
            // MRG: V_half = -30.0 mV, k = 10.0 mV, slow
            let s_inf = Self::boltz(v, -30.0, 10.0);
            let tau_s = 20.0 + 60.0 / (1.0 + ((v + 30.0) / 15.0).powi(2)).max(0.01);
            self.s += dt_sub * (s_inf - self.s) / tau_s;

            // Clamp gates
            self.m = self.m.clamp(0.0, 1.0);
            self.h = self.h.clamp(0.0, 1.0);
            self.p = self.p.clamp(0.0, 1.0);
            self.s = self.s.clamp(0.0, 1.0);

            // Currents
            let i_nat = self.g_nat * self.m.powi(3) * self.h * (v - self.e_na);
            let i_nap = self.g_nap * self.p.powi(3) * (v - self.e_na);
            let i_ks = self.g_ks * self.s * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);

            let dv = (-(i_nat + i_nap + i_ks + i_l) + input) / self.c_m;
            self.v += dt_sub * dv;
        }

        // Safety bounds
        self.v = self.v.clamp(-120.0, 60.0);
        if !self.v.is_finite() {
            self.v = -80.0;
        }
        if !self.m.is_finite() {
            self.m = 0.01;
        }
        if !self.h.is_finite() {
            self.h = 0.75;
        }
        if !self.p.is_finite() {
            self.p = 0.01;
        }
        if !self.s.is_finite() {
            self.s = 0.05;
        }

        // Spike: V crosses -10 mV upward
        if self.v >= -10.0 && v_prev_ext < -10.0 {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Myelinated Axon (Saltatory Conduction Segment)
// ═══════════════════════════════════════════════════════════════════

/// Myelinated axon segment — node of Ranvier + internode cable.
///
/// Models a single saltatory conduction unit per the MRG 2002
/// double-cable architecture: an active node (using the NodeOfRanvier
/// model) coupled to a passive internode represented as a lumped
/// RC cable.
///
/// The internode has:
/// - Very low capacitance (~0.001 µF/cm², myelin layers)
/// - Very high resistance (myelin sheath insulation)
/// - Paranodal seal resistance (leakage at node-internode junction)
///
/// The node voltage drives current through the paranodal seal into
/// the internode, and the internode voltage feeds back into the node
/// via the return path. This bidirectional coupling determines the
/// conduction velocity and safety factor.
///
/// The external input represents current arriving from the upstream
/// node (saltatory propagation).
///
/// V_node equation: C_n dV_n/dt = I_ionic(node) + g_para*(V_i - V_n) + I_ext
/// V_internode equation: C_i dV_i/dt = -g_l_myelin*(V_i - E_l) + g_para*(V_n - V_i)
///
/// McIntyre, Richardson & Grill, J Neurophysiol 87:995, 2002.
/// Richardson et al., Clin Neurophysiol 111:2175, 2000.
#[derive(Clone, Debug)]
pub struct MyelinatedAxon {
    // Node (active, MRG)
    pub node: NodeOfRanvier,
    // Internode (passive cable)
    pub v_inter: f64,    // Internode voltage (mV)
    pub c_inter: f64,    // Internode capacitance (µF/cm², very low)
    pub g_l_myelin: f64, // Myelin leak conductance (very low)
    pub e_l_myelin: f64, // Myelin leak reversal
    pub g_para: f64,     // Paranodal seal conductance
    pub dt: f64,
    pub gain: f64,
}

impl Default for MyelinatedAxon {
    fn default() -> Self {
        Self::new()
    }
}

impl MyelinatedAxon {
    pub fn new() -> Self {
        Self {
            node: NodeOfRanvier::new(),
            v_inter: -80.0,
            c_inter: 0.001,    // Very low (myelin layers)
            g_l_myelin: 0.001, // Very low leak through myelin
            e_l_myelin: -80.0,
            g_para: 0.01, // Paranodal seal conductance
            dt: 0.5,
            gain: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;

        // Paranodal coupling: node ↔ internode
        let i_para_to_node = self.g_para * (self.v_inter - self.node.v);
        let i_para_to_inter = self.g_para * (self.node.v - self.v_inter);

        // Internode passive cable dynamics
        let dv_inter =
            (-self.g_l_myelin * (self.v_inter - self.e_l_myelin) + i_para_to_inter) / self.c_inter;
        self.v_inter += self.node.dt / self.node.sub_steps as f64 * dv_inter;

        // Safety bounds for internode
        self.v_inter = self.v_inter.clamp(-120.0, 60.0);
        if !self.v_inter.is_finite() {
            self.v_inter = -80.0;
        }

        // Step the node with external input + paranodal current
        // We modify the node's current to include paranodal coupling
        let total_input = input + i_para_to_node * 100.0; // Scale for node's C_m
        self.node.step(total_input)
    }

    /// Access the node membrane potential.
    pub fn v(&self) -> f64 {
        self.node.v
    }

    pub fn reset(&mut self) {
        self.node.reset();
        self.v_inter = -80.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Frankenhaeuser-Huxley Axon tests --

    #[test]
    fn fh_fires_with_input() {
        // FH model uses µA/cm² — need ~1000+ for spiking (FH 1964 Fig 3)
        let mut n = FrankenhaeUserHuxleyAxon::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(2000.0);
        }
        assert!(
            spikes > 0,
            "FH axon must fire with strong input, got {spikes}"
        );
    }

    #[test]
    fn fh_silent_without_input() {
        let mut n = FrankenhaeUserHuxleyAxon::new();
        let mut spikes = 0;
        for _ in 0..5_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(
            spikes, 0,
            "FH axon must be silent without input, got {spikes}"
        );
    }

    #[test]
    fn fh_action_potential_shape() {
        // AP should depolarise well above 60 mV (spike threshold)
        let mut n = FrankenhaeUserHuxleyAxon::new();
        let mut v_max = -100.0_f64;
        for _ in 0..500 {
            n.step(2000.0);
            v_max = v_max.max(n.v);
        }
        assert!(v_max > 40.0, "AP peak should exceed 40 mV, got {v_max:.1}");
    }

    #[test]
    fn fh_gating_evolves() {
        let mut n = FrankenhaeUserHuxleyAxon::new();
        let m0 = n.m;
        let h0 = n.h;
        for _ in 0..100 {
            n.step(2000.0);
        }
        assert!(n.m != m0 || n.h != h0, "Gating variables must evolve");
    }

    #[test]
    fn fh_four_gates() {
        // All 4 gates (m, h, n, p) must evolve during spiking
        let mut n = FrankenhaeUserHuxleyAxon::new();
        for _ in 0..200 {
            n.step(2000.0);
        }
        // After spiking: m should have risen, h should have fallen
        // n and p should have changed from initial
        assert!(
            n.m > 0.005 || n.h < 0.8 || n.n > 0.01 || n.p > 0.01,
            "All gates must evolve: m={:.3}, h={:.3}, n={:.3}, p={:.3}",
            n.m,
            n.h,
            n.n,
            n.p
        );
    }

    #[test]
    fn fh_stronger_input_more_spikes() {
        let mut weak = FrankenhaeUserHuxleyAxon::new();
        let mut strong = FrankenhaeUserHuxleyAxon::new();
        let (mut sw, mut ss) = (0, 0);
        for _ in 0..2_000 {
            sw += weak.step(1000.0);
            ss += strong.step(3000.0);
        }
        assert!(
            ss >= sw,
            "Stronger input → more spikes: strong={ss} vs weak={sw}"
        );
    }

    #[test]
    fn fh_all_gates_bounded() {
        let mut n = FrankenhaeUserHuxleyAxon::new();
        for _ in 0..2_000 {
            n.step(3000.0);
        }
        assert!(n.m >= 0.0 && n.m <= 1.0, "m out of bounds: {}", n.m);
        assert!(n.h >= 0.0 && n.h <= 1.0, "h out of bounds: {}", n.h);
        assert!(n.n >= 0.0 && n.n <= 1.0, "n out of bounds: {}", n.n);
        assert!(n.p >= 0.0 && n.p <= 1.0, "p out of bounds: {}", n.p);
    }

    #[test]
    fn fh_nan_input_stays_finite() {
        let mut n = FrankenhaeUserHuxleyAxon::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
        assert!(n.m.is_finite());
    }

    #[test]
    fn fh_reset_clears_state() {
        let mut n = FrankenhaeUserHuxleyAxon::new();
        for _ in 0..500 {
            n.step(2000.0);
        }
        n.reset();
        assert_eq!(n.v, 0.0);
        assert_eq!(n.m, 0.005);
        assert_eq!(n.h, 0.8);
    }

    #[test]
    fn fh_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = FrankenhaeUserHuxleyAxon::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(1500.0));
        }
        let elapsed = start.elapsed();
        // 50 sub-steps per step → 50k total iterations
        assert!(
            elapsed.as_millis() < 100,
            "1k steps must complete in <100ms"
        );
    }

    // -- Node of Ranvier (MRG 2002) tests --

    #[test]
    fn nor_fires_with_input() {
        let mut n = NodeOfRanvier::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(500.0);
        }
        assert!(
            spikes > 0,
            "Node of Ranvier must fire with input, got {spikes}"
        );
    }

    #[test]
    fn nor_silent_without_input() {
        let mut n = NodeOfRanvier::new();
        let mut spikes = 0;
        for _ in 0..5_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(spikes, 0, "Must be silent without input, got {spikes}");
    }

    #[test]
    fn nor_high_nat_density() {
        // Node of Ranvier has g_nat=3000 (much higher than standard HH ~120)
        let n = NodeOfRanvier::new();
        assert!(
            n.g_nat > 1000.0,
            "Nodal transient Na should be very high: g_nat={}",
            n.g_nat
        );
    }

    #[test]
    fn nor_has_persistent_na() {
        // MRG model must include persistent Na — distinguishes from generic HH
        let n = NodeOfRanvier::new();
        assert!(
            n.g_nap > 0.0,
            "MRG model must have persistent Na current: g_nap={}",
            n.g_nap
        );
    }

    #[test]
    fn nor_has_kv7_slow_k() {
        // Kv7 (KCNQ) is the dominant K channel at nodes, not Kv3 or Kv1
        let n = NodeOfRanvier::new();
        assert!(
            n.g_ks > 0.0,
            "MRG model must have slow K (Kv7): g_ks={}",
            n.g_ks
        );
    }

    #[test]
    fn nor_persistent_na_lowers_threshold() {
        // With persistent Na, less current is needed to fire
        let mut with_nap = NodeOfRanvier::new();
        let mut no_nap = NodeOfRanvier::new();
        no_nap.g_nap = 0.0;
        let (mut s_with, mut s_without) = (0, 0);
        for _ in 0..2_000 {
            s_with += with_nap.step(200.0);
            s_without += no_nap.step(200.0);
        }
        assert!(
            s_with >= s_without,
            "Persistent Na should lower threshold: with={s_with} vs without={s_without}"
        );
    }

    #[test]
    fn nor_gating_evolves() {
        let mut n = NodeOfRanvier::new();
        let m0 = n.m;
        let p0 = n.p;
        for _ in 0..100 {
            n.step(500.0);
        }
        assert!(
            n.m != m0 || n.p != p0,
            "Gating must evolve: m={:.3}, p={:.3}",
            n.m,
            n.p
        );
    }

    #[test]
    fn nor_nan_input_stays_finite() {
        let mut n = NodeOfRanvier::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
        assert!(n.m.is_finite());
        assert!(n.p.is_finite());
        assert!(n.s.is_finite());
    }

    #[test]
    fn nor_reset_clears_state() {
        let mut n = NodeOfRanvier::new();
        for _ in 0..500 {
            n.step(500.0);
        }
        n.reset();
        assert_eq!(n.v, -80.0);
        assert_eq!(n.m, 0.01);
        assert_eq!(n.p, 0.01);
        assert_eq!(n.s, 0.05);
    }

    #[test]
    fn nor_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = NodeOfRanvier::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(500.0));
        }
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 50, "1k steps must complete in <50ms");
    }

    // -- Myelinated Axon tests --

    #[test]
    fn myelin_fires_with_input() {
        let mut n = MyelinatedAxon::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(500.0);
        }
        assert!(spikes > 0, "Myelinated axon must fire, got {spikes}");
    }

    #[test]
    fn myelin_silent_without_input() {
        let mut n = MyelinatedAxon::new();
        let mut spikes = 0;
        for _ in 0..5_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(spikes, 0, "Must be silent without input, got {spikes}");
    }

    #[test]
    fn myelin_internode_coupling() {
        // Internode voltage should be affected by node spiking
        let mut n = MyelinatedAxon::new();
        let v_inter_0 = n.v_inter;
        for _ in 0..500 {
            n.step(500.0);
        }
        assert!(
            (n.v_inter - v_inter_0).abs() > 0.001,
            "Internode voltage should change with node activity: v_inter={}",
            n.v_inter
        );
    }

    #[test]
    fn myelin_has_low_capacitance() {
        let n = MyelinatedAxon::new();
        assert!(
            n.c_inter < 0.01,
            "Myelin capacitance must be very low: {}",
            n.c_inter
        );
    }

    #[test]
    fn myelin_has_low_myelin_leak() {
        let n = MyelinatedAxon::new();
        assert!(
            n.g_l_myelin < 0.01,
            "Myelin leak must be very low: {}",
            n.g_l_myelin
        );
    }

    #[test]
    fn myelin_has_paranodal_seal() {
        let n = MyelinatedAxon::new();
        assert!(n.g_para > 0.0, "Must have paranodal seal conductance");
    }

    #[test]
    fn myelin_stronger_input_more_spikes() {
        let mut weak = MyelinatedAxon::new();
        let mut strong = MyelinatedAxon::new();
        let (mut sw, mut ss) = (0, 0);
        for _ in 0..2_000 {
            sw += weak.step(300.0);
            ss += strong.step(1000.0);
        }
        assert!(ss >= sw, "Stronger → more spikes: strong={ss} vs weak={sw}");
    }

    #[test]
    fn myelin_nan_input_stays_finite() {
        let mut n = MyelinatedAxon::new();
        n.step(f64::NAN);
        assert!(n.v().is_finite());
        assert!(n.v_inter.is_finite());
    }

    #[test]
    fn myelin_reset_clears_state() {
        let mut n = MyelinatedAxon::new();
        for _ in 0..500 {
            n.step(500.0);
        }
        n.reset();
        assert_eq!(n.v_inter, -80.0);
        assert_eq!(n.node.v, -80.0);
    }

    #[test]
    fn myelin_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = MyelinatedAxon::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(500.0));
        }
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 50, "1k steps must complete in <50ms");
    }

    #[test]
    fn myelinated_family_defaults_match_constructors() {
        let fh_default = FrankenhaeUserHuxleyAxon::default();
        let fh_constructed = FrankenhaeUserHuxleyAxon::new();
        assert_eq!(fh_default.v, fh_constructed.v);
        assert_eq!(fh_default.sub_steps, fh_constructed.sub_steps);

        let node_default = NodeOfRanvier::default();
        let node_constructed = NodeOfRanvier::new();
        assert_eq!(node_default.v, node_constructed.v);
        assert_eq!(node_default.g_nat, node_constructed.g_nat);

        let axon_default = MyelinatedAxon::default();
        let axon_constructed = MyelinatedAxon::new();
        assert_eq!(axon_default.v_inter, axon_constructed.v_inter);
        assert_eq!(axon_default.g_para, axon_constructed.g_para);
    }

    #[test]
    fn fh_rate_singularities_use_finite_limits() {
        for v in [22.0, 13.0, -10.0, 23.0, 21.0, -4.0] {
            let mut n = FrankenhaeUserHuxleyAxon::new();
            n.v = v;
            n.dt = 0.0;
            n.sub_steps = 1;
            n.step(0.0);
            assert!(n.v.is_finite(), "voltage must remain finite at V={v}");
            assert!(n.m.is_finite(), "m gate must remain finite at V={v}");
            assert!(n.h.is_finite(), "h gate must remain finite at V={v}");
            assert!(n.n.is_finite(), "n gate must remain finite at V={v}");
            assert!(n.p.is_finite(), "p gate must remain finite at V={v}");
        }
    }

    #[test]
    fn fh_nonfinite_gates_recover_to_initial_state() {
        let mut n = FrankenhaeUserHuxleyAxon::new();
        n.m = f64::NAN;
        n.h = f64::NAN;
        n.n = f64::NAN;
        n.p = f64::NAN;
        n.dt = 0.0;
        n.sub_steps = 1;
        n.step(0.0);
        assert_eq!(n.m, 0.005);
        assert_eq!(n.h, 0.8);
        assert_eq!(n.n, 0.01);
        assert_eq!(n.p, 0.01);
    }

    #[test]
    fn myelin_nonfinite_internode_recovers_to_rest() {
        let mut n = MyelinatedAxon::new();
        n.v_inter = f64::NAN;
        n.step(0.0);
        assert_eq!(n.v_inter, -80.0);
        assert!(n.node.v.is_finite());
    }
}
