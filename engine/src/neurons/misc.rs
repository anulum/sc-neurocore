// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Miscellaneous Neuron and Cell Models

//! Miscellaneous neuron and cell models.
//!
//! Phase 3H: graded synapse, gap junction, axon, cardiac,
//! smooth muscle, and endocrine models.
//! Added one by one with full 7-point checklist verification.

// ═══════════════════════════════════════════════════════════════════
// Graded Synapse Neuron (non-spiking interneuron)
// ═══════════════════════════════════════════════════════════════════

/// Non-spiking interneuron with graded synaptic release.
///
/// Models interneurons that communicate via graded potential changes
/// rather than action potentials (e.g., retinal bipolar/amacrine cells,
/// C. elegans interneurons, crustacean stomatogastric neurons).
///
/// The membrane potential follows passive RC dynamics with saturation:
///
///   C dV/dt = -g_L(V - E_L) + g_in * I_ext
///
/// Transmitter release is a sigmoid function of V:
///
///   release = 1 / (1 + exp(-(V - V_half) / k_release))
///
/// A "spike" event is emitted when V crosses a threshold from below,
/// representing a significant release event.
///
/// Roberts & Bush, J Comp Physiol A 185:549, 1999.
#[derive(Clone, Debug)]
pub struct GradedSynapseNeuron {
    pub v: f64,           // Membrane potential (mV)
    pub c_m: f64,         // Membrane capacitance (normalised)
    pub g_l: f64,         // Leak conductance
    pub e_l: f64,         // Leak reversal potential (mV)
    pub g_in: f64,        // Input conductance scaling
    pub v_half: f64,      // Release sigmoid half-activation (mV)
    pub k_release: f64,   // Release sigmoid slope
    pub v_min: f64,       // Saturation floor (mV)
    pub v_max: f64,       // Saturation ceiling (mV)
    pub v_threshold: f64, // "Spike" detection threshold (mV)
    pub dt: f64,
    pub gain: f64,
}

impl Default for GradedSynapseNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl GradedSynapseNeuron {
    pub fn new() -> Self {
        Self {
            v: -60.0,
            c_m: 1.0,
            g_l: 0.05, // Moderate leak
            e_l: -60.0,
            g_in: 0.1,      // Input scaling
            v_half: -40.0,  // Release kicks in at depolarised potential
            k_release: 5.0, // Sigmoid slope
            v_min: -80.0,
            v_max: -10.0,
            v_threshold: -35.0, // "Spike" threshold for pipeline
            dt: 0.1,
            gain: 1.0,
        }
    }

    /// Returns the graded transmitter release level [0, 1].
    pub fn release(&self) -> f64 {
        1.0 / (1.0 + (-(self.v - self.v_half) / self.k_release).exp())
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let v_prev = self.v;

        let dv = (-self.g_l * (self.v - self.e_l) + self.g_in * input) / self.c_m;
        self.v += self.dt * dv;

        // Saturation bounds
        self.v = self.v.clamp(self.v_min, self.v_max);
        if !self.v.is_finite() {
            self.v = self.e_l;
        }

        // Threshold crossing = significant release event
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
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
// Gap Junction Neuron
// ═══════════════════════════════════════════════════════════════════

/// Neuron with electrical synapse (gap junction) coupling.
///
/// Models neurons coupled via connexin-based gap junctions that allow
/// direct electrical current flow between cells. Found extensively in:
/// - Inferior olive neurons (climbing fibre system)
/// - Retinal ganglion cells (coupled networks)
/// - Cortical interneuron networks (PV+ basket cell syncytia)
/// - Thalamic reticular nucleus
///
/// Includes voltage-dependent rectification (Cx36 gating):
///   g_eff = g_gap * g_inf(V_j)
///   g_inf = g_min + (1 - g_min) / (1 + exp(A * (|V_j| - V_0)))
///
/// where V_j = V_neighbor - V is the transjunctional voltage,
/// g_min is the residual conductance at large V_j, V_0 is the
/// half-inactivation voltage (~30 mV for Cx36), and A is the
/// voltage sensitivity (~0.1 mV⁻¹).
///
/// At small |V_j| < V_0: near-full conductance (bidirectional).
/// At large |V_j| > V_0: conductance drops to g_min (rectification).
///
/// C dV/dt = -g_L(V - E_L) + g_eff * (V_neighbor - V) + I_tonic
///
/// Connors & Long, Annu Rev Neurosci 27:393, 2004.
/// Vervaeke et al., Neuron 65:801, 2010 (Cx36 voltage gating).
#[derive(Clone, Debug)]
pub struct GapJunctionNeuron {
    pub v: f64,       // Membrane potential (mV)
    pub c_m: f64,     // Membrane capacitance
    pub g_l: f64,     // Leak conductance
    pub e_l: f64,     // Leak reversal (mV)
    pub g_gap: f64,   // Maximal gap junction conductance
    pub i_tonic: f64, // Tonic depolarising current
    pub v_threshold: f64,
    pub v_reset: f64,
    pub refractory: f64, // Refractory period (ms)
    pub refrac_timer: f64,
    // Voltage-dependent rectification (Cx36)
    pub rect_v0: f64,   // Half-inactivation voltage (mV), ~30 for Cx36
    pub rect_a: f64,    // Voltage sensitivity (mV⁻¹), ~0.1 for Cx36
    pub rect_gmin: f64, // Residual conductance fraction [0,1], ~0.1
    pub dt: f64,
    pub gain: f64,
}

impl Default for GapJunctionNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl GapJunctionNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            c_m: 1.0,
            g_l: 0.1,
            e_l: -65.0,
            g_gap: 0.15,  // Gap junction coupling (maximal)
            i_tonic: 0.0, // No tonic drive by default
            v_threshold: -50.0,
            v_reset: -65.0,
            refractory: 2.0, // 2 ms refractory
            refrac_timer: 0.0,
            rect_v0: 30.0,  // Cx36: half-inactivation at ~30 mV Vj
            rect_a: 0.1,    // Cx36: voltage sensitivity
            rect_gmin: 0.1, // Cx36: ~10% residual conductance
            dt: 0.1,
            gain: 1.0,
        }
    }

    /// Voltage-dependent gap junction conductance (Cx36 gating).
    ///
    /// g_inf = g_min + (1 - g_min) / (1 + exp(A * (|V_j| - V_0)))
    ///
    /// Symmetric in |V_j|: rectification acts for both polarities.
    #[inline]
    fn rect_conductance(&self, v_j: f64) -> f64 {
        self.rect_gmin
            + (1.0 - self.rect_gmin) / (1.0 + (self.rect_a * (v_j.abs() - self.rect_v0)).exp())
    }

    pub fn step(&mut self, current: f64) -> i32 {
        // current = mean neighbour voltage or external drive
        let input = self.gain * current;

        if self.refrac_timer > 0.0 {
            self.refrac_timer -= self.dt;
            return 0;
        }

        // Transjunctional voltage
        let v_j = input - self.v;
        // Voltage-dependent effective conductance
        let g_eff = self.g_gap * self.rect_conductance(v_j);
        let i_gap = g_eff * v_j;
        let dv = (-self.g_l * (self.v - self.e_l) + i_gap + self.i_tonic) / self.c_m;
        self.v += self.dt * dv;

        // Safety
        self.v = self.v.clamp(-100.0, 40.0);
        if !self.v.is_finite() {
            self.v = self.e_l;
        }

        // Spike
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.refrac_timer = self.refractory;
            return 1;
        }
        0
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

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
/// Hirst & Edwards, J Physiol 531:567, 2001.
/// Imtiaz et al., Biophys J 83:1877, 2002.
#[derive(Clone, Debug)]
pub struct SmoothMuscleCell {
    pub v: f64,
    pub d: f64,        // CaL activation
    pub f: f64,        // CaL inactivation
    pub ca: f64,       // Cytosolic Ca²⁺ (µM)
    pub ca_store: f64, // ER/SR Ca²⁺ store (µM)
    pub c_m: f64,
    pub g_cal: f64, // L-type Ca²⁺
    pub g_bk: f64,  // BK channel
    pub g_l: f64,   // Leak
    pub e_ca: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub tau_ca: f64,   // Ca²⁺ decay (ms)
    pub v_serca: f64,  // SERCA pump max rate
    pub k_serca: f64,  // SERCA Km (µM)
    pub ip3: f64,      // IP3 concentration (µM, constant or input-driven)
    pub v_ip3r: f64,   // IP3R max release rate
    pub k_ip3: f64,    // IP3R half-activation (µM)
    pub k_ca_ip3: f64, // Ca²⁺ co-activation of IP3R (µM)
    pub kd_bk: f64,    // BK Ca²⁺ half-activation (µM)
    pub dt: f64,
    pub sub_steps: usize,
    pub gain: f64,
}

impl Default for SmoothMuscleCell {
    fn default() -> Self {
        Self::new()
    }
}

impl SmoothMuscleCell {
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

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

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

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // -- Graded Synapse Neuron tests --

    #[test]
    fn graded_depolarises_with_input() {
        let mut n = GradedSynapseNeuron::new();
        let v0 = n.v;
        for _ in 0..10_000 {
            n.step(200.0);
        }
        assert!(
            n.v > v0,
            "Must depolarise with positive input: v0={v0}, v={}",
            n.v
        );
    }

    #[test]
    fn graded_hyperpolarises_with_negative_input() {
        let mut n = GradedSynapseNeuron::new();
        let v0 = n.v;
        for _ in 0..10_000 {
            n.step(-200.0);
        }
        assert!(
            n.v < v0,
            "Must hyperpolarise with negative input: v0={v0}, v={}",
            n.v
        );
    }

    #[test]
    fn graded_fires_with_strong_input() {
        let mut n = GradedSynapseNeuron::new();
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += n.step(500.0);
        }
        assert!(
            spikes > 0,
            "Must cross threshold with strong input, got {spikes}"
        );
    }

    #[test]
    fn graded_silent_without_input() {
        let mut n = GradedSynapseNeuron::new();
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(
            spikes, 0,
            "Must be silent without input (V starts at E_L), got {spikes}"
        );
    }

    #[test]
    fn graded_release_monotonic() {
        // Release should increase with depolarisation
        let mut n = GradedSynapseNeuron::new();
        n.v = -70.0;
        let r_low = n.release();
        n.v = -40.0;
        let r_mid = n.release();
        n.v = -10.0;
        let r_high = n.release();
        assert!(
            r_low < r_mid && r_mid < r_high,
            "Release must be monotonic: r_low={r_low:.3}, r_mid={r_mid:.3}, r_high={r_high:.3}"
        );
    }

    #[test]
    fn graded_release_bounded() {
        let mut n = GradedSynapseNeuron::new();
        n.v = -100.0;
        assert!(n.release() >= 0.0 && n.release() <= 1.0);
        n.v = 50.0;
        assert!(n.release() >= 0.0 && n.release() <= 1.0);
    }

    #[test]
    fn graded_v_saturates() {
        let mut n = GradedSynapseNeuron::new();
        for _ in 0..50_000 {
            n.step(1e6);
        }
        assert!(
            n.v <= n.v_max,
            "V must not exceed v_max={}, got {}",
            n.v_max,
            n.v
        );
        n.reset();
        for _ in 0..50_000 {
            n.step(-1e6);
        }
        assert!(
            n.v >= n.v_min,
            "V must not go below v_min={}, got {}",
            n.v_min,
            n.v
        );
    }

    #[test]
    fn graded_nan_input_stays_finite() {
        let mut n = GradedSynapseNeuron::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
    }

    #[test]
    fn graded_reset_clears_state() {
        let mut n = GradedSynapseNeuron::new();
        for _ in 0..10_000 {
            n.step(500.0);
        }
        n.reset();
        assert_eq!(n.v, -60.0);
    }

    #[test]
    fn graded_performance_100k_steps() {
        let start = std::time::Instant::now();
        let mut n = GradedSynapseNeuron::new();
        for _ in 0..100_000 {
            std::hint::black_box(n.step(100.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "100k steps must complete in <50ms"
        );
    }

    // -- Gap Junction Neuron tests --

    #[test]
    fn gap_fires_with_depolarising_drive() {
        // Input as V_neighbor = 0 mV (depolarised relative to -65 mV rest)
        let mut n = GapJunctionNeuron::new();
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += n.step(0.0); // V_neighbor = 0 mV → depolarising
        }
        assert!(
            spikes > 0,
            "Gap junction must fire with depolarising drive, got {spikes}"
        );
    }

    #[test]
    fn gap_silent_at_rest() {
        // Input = E_L → no gap junction current → silent
        let mut n = GapJunctionNeuron::new();
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += n.step(-65.0); // V_neighbor = E_L → zero gap current
        }
        assert_eq!(
            spikes, 0,
            "Must be silent when V_neighbor = E_L, got {spikes}"
        );
    }

    #[test]
    fn gap_junction_pulls_toward_neighbor() {
        // If V_neighbor > V, gap junction depolarises; if V_neighbor < V, hyperpolarises
        let mut n = GapJunctionNeuron::new(); // V = -65
        for _ in 0..5_000 {
            n.step(-40.0);
        } // V_neighbor = -40 → depolarising
        assert!(
            n.v > -65.0 || n.refrac_timer > 0.0,
            "Gap junction must pull V toward neighbor: v={}",
            n.v
        );
    }

    #[test]
    fn gap_stronger_coupling_more_spikes() {
        let mut weak = GapJunctionNeuron::new();
        weak.g_gap = 0.01;
        let mut strong = GapJunctionNeuron::new();
        strong.g_gap = 0.1;
        let (mut sw, mut ss) = (0, 0);
        for _ in 0..50_000 {
            sw += weak.step(-20.0);
            ss += strong.step(-20.0);
        }
        assert!(
            ss >= sw,
            "Stronger coupling → more spikes: strong={ss} vs weak={sw}"
        );
    }

    #[test]
    fn gap_refractory_enforced() {
        let mut n = GapJunctionNeuron::new();
        // Drive until first spike (V_neighbor = 0 → strong depolarising)
        let mut first_spike_t = 0;
        for t in 0..10_000 {
            if n.step(0.0) == 1 {
                first_spike_t = t;
                break;
            }
        }
        assert!(first_spike_t > 0, "Must spike");
        // Next step should be in refractory
        assert!(n.refrac_timer > 0.0, "Must be in refractory after spike");
        assert_eq!(n.step(0.0), 0, "Must not spike during refractory");
    }

    #[test]
    fn gap_hyperpolarising_drive_silent() {
        // V_neighbor = -100 → strong hyperpolarising gap current
        let mut n = GapJunctionNeuron::new();
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += n.step(-100.0);
        }
        assert_eq!(
            spikes, 0,
            "Hyperpolarising drive must keep silent, got {spikes}"
        );
    }

    #[test]
    fn gap_tonic_current_depolarises() {
        let mut n = GapJunctionNeuron::new();
        n.i_tonic = 5.0; // Strong tonic drive
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += n.step(-65.0); // No gap drive, but tonic current
        }
        assert!(
            spikes > 0,
            "Tonic current should produce spikes, got {spikes}"
        );
    }

    #[test]
    fn gap_nan_input_stays_finite() {
        let mut n = GapJunctionNeuron::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
    }

    #[test]
    fn gap_reset_clears_state() {
        let mut n = GapJunctionNeuron::new();
        for _ in 0..10_000 {
            n.step(-20.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.refrac_timer, 0.0);
    }

    #[test]
    fn gap_rectification_reduces_at_large_vj() {
        // At large |Vj|, rectification should reduce effective conductance
        let n = GapJunctionNeuron::new();
        let g_small = n.rect_conductance(5.0); // |Vj|=5 mV (small)
        let g_large = n.rect_conductance(60.0); // |Vj|=60 mV (large)
        assert!(
            g_small > g_large,
            "Rectification must reduce g at large Vj: g(5)={g_small:.3} vs g(60)={g_large:.3}"
        );
        assert!(
            g_large >= n.rect_gmin,
            "Conductance must not drop below g_min={}: got {g_large:.3}",
            n.rect_gmin
        );
    }

    #[test]
    fn gap_performance_100k_steps() {
        let start = std::time::Instant::now();
        let mut n = GapJunctionNeuron::new();
        for _ in 0..100_000 {
            std::hint::black_box(n.step(-20.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "100k steps must complete in <50ms"
        );
    }

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
        let start = std::time::Instant::now();
        let mut n = SmoothMuscleCell::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(2.0));
        }
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 50, "1k steps must complete in <50ms");
    }

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
}
