// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Gap Junction Neuron Model

//! Electrical-synapse neuron with voltage-dependent gap-junction coupling.

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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn gap_default_matches_constructor() {
        let default = GapJunctionNeuron::default();
        let constructed = GapJunctionNeuron::new();
        assert_eq!(default.v, constructed.v);
        assert_eq!(default.g_gap, constructed.g_gap);
        assert_eq!(default.rect_v0, constructed.rect_v0);
    }
}
