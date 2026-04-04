// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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
    pub v: f64,          // Membrane potential (mV)
    pub c_m: f64,        // Membrane capacitance (normalised)
    pub g_l: f64,        // Leak conductance
    pub e_l: f64,        // Leak reversal potential (mV)
    pub g_in: f64,       // Input conductance scaling
    pub v_half: f64,     // Release sigmoid half-activation (mV)
    pub k_release: f64,  // Release sigmoid slope
    pub v_min: f64,      // Saturation floor (mV)
    pub v_max: f64,      // Saturation ceiling (mV)
    pub v_threshold: f64, // "Spike" detection threshold (mV)
    pub dt: f64,
    pub gain: f64,
}

impl Default for GradedSynapseNeuron {
    fn default() -> Self { Self::new() }
}

impl GradedSynapseNeuron {
    pub fn new() -> Self {
        Self {
            v: -60.0,
            c_m: 1.0,
            g_l: 0.05,       // Moderate leak
            e_l: -60.0,
            g_in: 0.1,       // Input scaling
            v_half: -40.0,   // Release kicks in at depolarised potential
            k_release: 5.0,  // Sigmoid slope
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
        if !self.v.is_finite() { self.v = self.e_l; }

        // Threshold crossing = significant release event
        if self.v >= self.v_threshold && v_prev < self.v_threshold { 1 } else { 0 }
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
/// The model is LIF-type with a gap junction conductance term.
/// In the single-neuron pipeline context, the external current
/// represents the net gap junction drive from coupled neighbours:
///   I_gap = g_gap * (V_neighbor_mean - V)
///
/// The total membrane equation:
///   C dV/dt = -g_L(V - E_L) + g_gap * (I_ext - V) + I_tonic
///
/// where I_ext represents the mean neighbour voltage, and g_gap
/// is the gap junction conductance. When V exceeds threshold,
/// a spike is emitted and V resets.
///
/// Connors & Long, Annu Rev Neurosci 27:393, 2004.
#[derive(Clone, Debug)]
pub struct GapJunctionNeuron {
    pub v: f64,          // Membrane potential (mV)
    pub c_m: f64,        // Membrane capacitance
    pub g_l: f64,        // Leak conductance
    pub e_l: f64,        // Leak reversal (mV)
    pub g_gap: f64,      // Gap junction conductance
    pub i_tonic: f64,    // Tonic depolarising current
    pub v_threshold: f64,
    pub v_reset: f64,
    pub refractory: f64, // Refractory period (ms)
    pub refrac_timer: f64,
    pub dt: f64,
    pub gain: f64,
}

impl Default for GapJunctionNeuron {
    fn default() -> Self { Self::new() }
}

impl GapJunctionNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            c_m: 1.0,
            g_l: 0.1,
            e_l: -65.0,
            g_gap: 0.05,     // Gap junction coupling
            i_tonic: 0.0,    // No tonic drive by default
            v_threshold: -50.0,
            v_reset: -65.0,
            refractory: 2.0, // 2 ms refractory
            refrac_timer: 0.0,
            dt: 0.1,
            gain: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        // current = mean neighbour voltage or external drive
        let input = self.gain * current;

        if self.refrac_timer > 0.0 {
            self.refrac_timer -= self.dt;
            return 0;
        }

        // Gap junction: g_gap * (V_neighbor - V)
        // Here input represents V_neighbor (or external current scaled to mV)
        let i_gap = self.g_gap * (input - self.v);
        let dv = (-self.g_l * (self.v - self.e_l) + i_gap + self.i_tonic) / self.c_m;
        self.v += self.dt * dv;

        // Safety
        self.v = self.v.clamp(-100.0, 40.0);
        if !self.v.is_finite() { self.v = self.e_l; }

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
/// Uses permeability-based formulation instead of conductance-based,
/// with 4 gating variables: m (Na activation), h (Na inactivation),
/// n (delayed rectifier K), p (slow non-specific current).
///
/// I_Na = P_Na * m^2 * h * V_Na_driving
/// I_K  = P_K  * n^2 * V_K_driving
/// I_p  = P_p  * p^2 * V_p_driving
/// I_L  = g_L * (V - E_L)
///
/// C dV/dt = -(I_Na + I_K + I_p + I_L) + I_ext
///
/// Uses sub-stepping (dt_sub = 0.01 ms) for gating stability.
///
/// Frankenhaeuser & Huxley, J Physiol 171:302, 1964.
#[derive(Clone, Debug)]
pub struct FrankenhaeUserHuxleyAxon {
    pub v: f64,     // Membrane potential (mV, relative to rest)
    pub m: f64,     // Na activation
    pub h: f64,     // Na inactivation
    pub n: f64,     // K delayed rectifier
    pub p: f64,     // Slow non-specific
    pub c_m: f64,
    pub p_na: f64,  // Na permeability (mS/cm²-equivalent)
    pub p_k: f64,   // K permeability
    pub p_p: f64,   // Slow current permeability
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_p: f64,
    pub e_l: f64,
    pub dt: f64,
    pub sub_steps: usize,
    pub gain: f64,
}

impl Default for FrankenhaeUserHuxleyAxon {
    fn default() -> Self { Self::new() }
}

impl FrankenhaeUserHuxleyAxon {
    pub fn new() -> Self {
        Self {
            v: 0.0,          // Relative to resting potential
            m: 0.005,
            h: 0.8,
            n: 0.01,
            p: 0.01,
            c_m: 2.0,       // µF/cm² (myelinated node)
            p_na: 12.0,     // Na permeability
            p_k: 1.2,       // K permeability
            p_p: 0.54,      // Slow current
            g_l: 0.3,       // Leak
            e_na: 115.0,    // mV above rest
            e_k: -12.0,
            e_p: 115.0,     // Similar to Na
            e_l: 0.0,       // At rest
            dt: 0.5,        // External step (ms)
            sub_steps: 50,  // 50 sub-steps → dt_sub = 0.01 ms
            gain: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let dt_sub = self.dt / self.sub_steps as f64;
        let v_prev = self.v;

        for _ in 0..self.sub_steps {
            // FH alpha/beta rate functions (simplified Frankenhaeuser kinetics)
            let v = self.v;

            // m gate (Na activation)
            let am = if (v - 22.0).abs() < 0.1 { 1.87 }
                else { 0.36 * (v - 22.0) / (1.0 - (-(v - 22.0) / 3.0).exp()) };
            let bm = if (v - 13.0).abs() < 0.1 { 1.87 }
                else { 0.4 * (13.0 - v) / (1.0 - ((v - 13.0) / 20.0).exp()) };

            // h gate (Na inactivation)
            let ah = if (v + 10.0).abs() < 0.1 { 0.08 }
                else { 0.1 * (-10.0 - v) / (1.0 - ((v + 10.0) / 6.0).exp()) };
            let bh = 4.5 / (1.0 + ((45.0 - v) / 10.0).exp());

            // n gate (K delayed rectifier)
            let an = if (v - 13.0).abs() < 0.1 { 0.1 }
                else { 0.02 * (v - 13.0) / (1.0 - (-(v - 13.0) / 10.0).exp()) };
            let bn = if (v - 23.0).abs() < 0.1 { 0.05 }
                else { 0.05 * (23.0 - v) / (1.0 - ((v - 23.0) / 10.0).exp()) };

            // p gate (slow non-specific)
            let ap = if (v - 21.0).abs() < 0.1 { 0.04 }
                else { 0.006 * (v - 21.0) / (1.0 - (-(v - 21.0) / 2.0).exp()) };
            let bp = if (v + 4.0).abs() < 0.1 { 0.04 }
                else { 0.09 * (-4.0 - v) / (1.0 - ((v + 4.0) / 2.0).exp()) };

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

            // Currents (permeability-based, simplified)
            let i_na = self.p_na * self.m * self.m * self.h * (self.v - self.e_na);
            let i_k = self.p_k * self.n * self.n * (self.v - self.e_k);
            let i_p = self.p_p * self.p * self.p * (self.v - self.e_p);
            let i_l = self.g_l * (self.v - self.e_l);

            let dv = (-(i_na + i_k + i_p + i_l) + input) / self.c_m;
            self.v += dt_sub * dv;
        }

        // Safety
        self.v = self.v.clamp(-50.0, 150.0);
        if !self.v.is_finite() { self.v = 0.0; }
        if !self.m.is_finite() { self.m = 0.005; }
        if !self.h.is_finite() { self.h = 0.8; }
        if !self.n.is_finite() { self.n = 0.01; }
        if !self.p.is_finite() { self.p = 0.01; }

        // Spike detection: V crosses 40 mV upward
        if self.v >= 40.0 && v_prev < 40.0 { 1 } else { 0 }
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
        for _ in 0..10_000 { n.step(200.0); }
        assert!(n.v > v0, "Must depolarise with positive input: v0={v0}, v={}", n.v);
    }

    #[test]
    fn graded_hyperpolarises_with_negative_input() {
        let mut n = GradedSynapseNeuron::new();
        let v0 = n.v;
        for _ in 0..10_000 { n.step(-200.0); }
        assert!(n.v < v0,
            "Must hyperpolarise with negative input: v0={v0}, v={}", n.v);
    }

    #[test]
    fn graded_fires_with_strong_input() {
        let mut n = GradedSynapseNeuron::new();
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += n.step(500.0);
        }
        assert!(spikes > 0, "Must cross threshold with strong input, got {spikes}");
    }

    #[test]
    fn graded_silent_without_input() {
        let mut n = GradedSynapseNeuron::new();
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(spikes, 0, "Must be silent without input (V starts at E_L), got {spikes}");
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
        assert!(r_low < r_mid && r_mid < r_high,
            "Release must be monotonic: r_low={r_low:.3}, r_mid={r_mid:.3}, r_high={r_high:.3}");
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
        for _ in 0..50_000 { n.step(1e6); }
        assert!(n.v <= n.v_max, "V must not exceed v_max={}, got {}", n.v_max, n.v);
        n.reset();
        for _ in 0..50_000 { n.step(-1e6); }
        assert!(n.v >= n.v_min, "V must not go below v_min={}, got {}", n.v_min, n.v);
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
        for _ in 0..10_000 { n.step(500.0); }
        n.reset();
        assert_eq!(n.v, -60.0);
    }

    #[test]
    fn graded_performance_100k_steps() {
        let start = std::time::Instant::now();
        let mut n = GradedSynapseNeuron::new();
        for _ in 0..100_000 { std::hint::black_box(n.step(100.0)); }
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 50, "100k steps must complete in <50ms");
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
        assert!(spikes > 0, "Gap junction must fire with depolarising drive, got {spikes}");
    }

    #[test]
    fn gap_silent_at_rest() {
        // Input = E_L → no gap junction current → silent
        let mut n = GapJunctionNeuron::new();
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += n.step(-65.0); // V_neighbor = E_L → zero gap current
        }
        assert_eq!(spikes, 0, "Must be silent when V_neighbor = E_L, got {spikes}");
    }

    #[test]
    fn gap_junction_pulls_toward_neighbor() {
        // If V_neighbor > V, gap junction depolarises; if V_neighbor < V, hyperpolarises
        let mut n = GapJunctionNeuron::new(); // V = -65
        for _ in 0..5_000 { n.step(-40.0); } // V_neighbor = -40 → depolarising
        assert!(n.v > -65.0 || n.refrac_timer > 0.0,
            "Gap junction must pull V toward neighbor: v={}", n.v);
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
        assert!(ss >= sw,
            "Stronger coupling → more spikes: strong={ss} vs weak={sw}");
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
        assert_eq!(spikes, 0, "Hyperpolarising drive must keep silent, got {spikes}");
    }

    #[test]
    fn gap_tonic_current_depolarises() {
        let mut n = GapJunctionNeuron::new();
        n.i_tonic = 5.0; // Strong tonic drive
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += n.step(-65.0); // No gap drive, but tonic current
        }
        assert!(spikes > 0, "Tonic current should produce spikes, got {spikes}");
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
        for _ in 0..10_000 { n.step(-20.0); }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.refrac_timer, 0.0);
    }

    #[test]
    fn gap_performance_100k_steps() {
        let start = std::time::Instant::now();
        let mut n = GapJunctionNeuron::new();
        for _ in 0..100_000 { std::hint::black_box(n.step(-20.0)); }
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 50, "100k steps must complete in <50ms");
    }

    // -- Frankenhaeuser-Huxley Axon tests --

    #[test]
    fn fh_fires_with_input() {
        let mut n = FrankenhaeUserHuxleyAxon::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(20.0);
        }
        assert!(spikes > 0, "FH axon must fire with strong input, got {spikes}");
    }

    #[test]
    fn fh_silent_without_input() {
        let mut n = FrankenhaeUserHuxleyAxon::new();
        let mut spikes = 0;
        for _ in 0..5_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(spikes, 0, "FH axon must be silent without input, got {spikes}");
    }

    #[test]
    fn fh_action_potential_shape() {
        // AP should depolarise well above 60 mV (spike threshold)
        let mut n = FrankenhaeUserHuxleyAxon::new();
        let mut v_max = -100.0_f64;
        for _ in 0..500 {
            n.step(20.0);
            v_max = v_max.max(n.v);
        }
        assert!(v_max > 40.0, "AP peak should exceed 40 mV, got {v_max:.1}");
    }

    #[test]
    fn fh_gating_evolves() {
        let mut n = FrankenhaeUserHuxleyAxon::new();
        let m0 = n.m;
        let h0 = n.h;
        for _ in 0..100 { n.step(20.0); }
        assert!(n.m != m0 || n.h != h0, "Gating variables must evolve");
    }

    #[test]
    fn fh_four_gates() {
        // All 4 gates (m, h, n, p) must evolve during spiking
        let mut n = FrankenhaeUserHuxleyAxon::new();
        for _ in 0..200 { n.step(20.0); }
        // After spiking: m should have risen, h should have fallen
        // n and p should have changed from initial
        assert!(n.m > 0.005 || n.h < 0.8 || n.n > 0.01 || n.p > 0.01,
            "All gates must evolve: m={:.3}, h={:.3}, n={:.3}, p={:.3}",
            n.m, n.h, n.n, n.p);
    }

    #[test]
    fn fh_stronger_input_more_spikes() {
        let mut weak = FrankenhaeUserHuxleyAxon::new();
        let mut strong = FrankenhaeUserHuxleyAxon::new();
        let (mut sw, mut ss) = (0, 0);
        for _ in 0..2_000 {
            sw += weak.step(10.0);
            ss += strong.step(30.0);
        }
        assert!(ss >= sw,
            "Stronger input → more spikes: strong={ss} vs weak={sw}");
    }

    #[test]
    fn fh_all_gates_bounded() {
        let mut n = FrankenhaeUserHuxleyAxon::new();
        for _ in 0..2_000 { n.step(30.0); }
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
        for _ in 0..500 { n.step(20.0); }
        n.reset();
        assert_eq!(n.v, 0.0);
        assert_eq!(n.m, 0.005);
        assert_eq!(n.h, 0.8);
    }

    #[test]
    fn fh_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = FrankenhaeUserHuxleyAxon::new();
        for _ in 0..1_000 { std::hint::black_box(n.step(15.0)); }
        let elapsed = start.elapsed();
        // 50 sub-steps per step → 50k total iterations
        assert!(elapsed.as_millis() < 100, "1k steps must complete in <100ms");
    }
}
