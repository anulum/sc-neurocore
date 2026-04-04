// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Cerebellar Circuit Neuron Models

//! Cerebellar circuit neuron models for granular and molecular layer computations.
//!
//! Phase 3D: granule cell, Golgi cell, stellate cell, Lugaro cell,
//! unipolar brush cell, deep cerebellar nuclei neuron.
//! Added one by one with full 7-point checklist verification.

// ═══════════════════════════════════════════════════════════════════
// Granule Cell
// ═══════════════════════════════════════════════════════════════════

/// Cerebellar granule cell — most numerous neuron in the brain (~50%).
///
/// Biophysics: LIF core with tonic GABAergic inhibition from Golgi cells,
/// T-type Ca2+ current for post-inhibitory rebound bursting, and very high
/// input resistance due to tiny soma (6-8 µm). Four short dendrites receive
/// mossy fibre input at glomeruli; output via parallel fibres to Purkinje cells.
///
/// The tonic GABA conductance models the continuous inhibitory tone that
/// Golgi cells impose, keeping granule cells near threshold but rarely
/// spontaneously active. Release from inhibition (disinhibition) triggers
/// rebound bursts via T-type Ca2+ channels.
///
/// D'Angelo et al., J Neurosci 21(3), 2001; Bhalla & Bhatt, Cerebellum, 2012.
#[derive(Clone, Debug)]
pub struct GranuleCell {
    pub v: f64,
    // T-type Ca2+ gating
    pub s: f64,         // T-type inactivation (slow)
    // Conductances (mS/cm²)
    pub g_l: f64,       // Leak
    pub g_tonic: f64,   // Tonic GABA conductance
    pub g_t: f64,       // T-type Ca2+ conductance
    // Reversal potentials (mV)
    pub e_l: f64,
    pub e_gaba: f64,    // GABA reversal (-75 mV, shunting)
    pub e_ca: f64,      // Ca2+ reversal
    // Membrane
    pub tau_m: f64,     // Membrane time constant (ms) — very short for tiny soma
    pub c_m: f64,       // Specific capacitance (µF/cm²)
    pub v_threshold: f64,
    pub v_reset: f64,
    pub refrac_count: f64,
    pub refrac_period: f64,
    pub gain: f64,      // Input scaling
    pub dt: f64,
}

impl Default for GranuleCell {
    fn default() -> Self { Self::new() }
}

impl GranuleCell {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            s: 0.95,            // T-type inactivation de-inactivated at rest
            g_l: 0.05,          // Low leak for high Rin
            g_tonic: 0.02,      // Tonic GABA
            g_t: 0.03,          // T-type Ca2+
            e_l: -70.0,
            e_gaba: -75.0,
            e_ca: 120.0,
            tau_m: 5.0,         // Short tau for tiny soma
            c_m: 1.0,
            v_threshold: -40.0,
            v_reset: -70.0,
            refrac_count: 0.0,
            refrac_period: 1.0, // 1 ms refractory
            gain: 1.5,
            dt: 0.5,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        // Refractory period
        if self.refrac_count > 0.0 {
            self.refrac_count -= self.dt;
            self.v = self.v_reset;
            // T-type inactivation still evolves during refractory
            let s_inf = 1.0 / (1.0 + ((self.v + 60.0) / 6.5).exp());
            let tau_s = 20.0 + 50.0 / (1.0 + ((self.v + 65.0) / 10.0).exp());
            self.s += self.dt * (s_inf - self.s) / tau_s;
            return 0;
        }

        // T-type Ca2+ activation (fast, instantaneous steady-state)
        let m_t_inf = 1.0 / (1.0 + (-(self.v + 52.0) / 5.0).exp());
        // T-type Ca2+ inactivation (slow)
        let s_inf = 1.0 / (1.0 + ((self.v + 60.0) / 6.5).exp());
        let tau_s = 20.0 + 50.0 / (1.0 + ((self.v + 65.0) / 10.0).exp());

        // Currents
        let i_l = self.g_l * (self.v - self.e_l);
        let i_tonic = self.g_tonic * (self.v - self.e_gaba);
        let i_t = self.g_t * m_t_inf * m_t_inf * self.s * (self.v - self.e_ca);
        let i_ext = self.gain * current.max(0.0);

        // Membrane equation
        let dv = (-i_l - i_tonic - i_t + i_ext) / self.c_m;
        self.v += self.dt * dv / self.tau_m;

        // T-type inactivation update
        self.s += self.dt * (s_inf - self.s) / tau_s;

        // Spike detection
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.refrac_count = self.refrac_period;
            // Spike inactivates T-type channels
            self.s *= 0.5;
            return 1;
        }

        // Bound membrane potential
        if self.v < -100.0 { self.v = -100.0; }
        if self.v > 60.0 { self.v = 60.0; }
        if !self.v.is_finite() { self.v = self.v_reset; }
        if !self.s.is_finite() { self.s = 0.95; }

        0
    }

    pub fn reset(&mut self) {
        self.v = -70.0;
        self.s = 0.95;
        self.refrac_count = 0.0;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // -- Granule Cell tests --

    #[test]
    fn granule_fires_with_strong_input() {
        let mut n = GranuleCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(15.0);
        }
        assert!(spikes > 10, "Granule cell must fire with strong excitatory input, got {spikes}");
    }

    #[test]
    fn granule_silent_at_rest() {
        let mut n = GranuleCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(spikes, 0, "Granule cell must be silent without input (tonic GABA inhibition)");
    }

    #[test]
    fn granule_no_fire_weak_input() {
        // Tonic GABA raises effective threshold
        let mut n = GranuleCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(1.0);
        }
        assert!(spikes == 0, "Weak input should not overcome tonic GABA, got {spikes}");
    }

    #[test]
    fn granule_tonic_gaba_raises_threshold() {
        // Compare firing with and without tonic GABA
        let mut with_gaba = GranuleCell::new();
        let mut no_gaba = GranuleCell::new();
        no_gaba.g_tonic = 0.0;

        let input = 8.0;
        let mut spikes_gaba = 0;
        let mut spikes_no_gaba = 0;
        for _ in 0..10_000 {
            spikes_gaba += with_gaba.step(input);
            spikes_no_gaba += no_gaba.step(input);
        }
        assert!(
            spikes_no_gaba > spikes_gaba,
            "Removing tonic GABA must increase firing: no_gaba={spikes_no_gaba} vs gaba={spikes_gaba}"
        );
    }

    #[test]
    fn granule_rebound_burst() {
        // Release from hyperpolarisation triggers T-type rebound
        let mut n = GranuleCell::new();
        // Hyperpolarise to de-inactivate T-type channels
        for _ in 0..2000 {
            n.step(0.0);
        }
        // Ensure s is high (de-inactivated)
        assert!(n.s > 0.8, "T-type must be de-inactivated at rest, s={}", n.s);

        // Now provide input — T-type should help fire
        let mut spikes_early = 0;
        for _ in 0..200 {
            spikes_early += n.step(10.0);
        }

        // Compare with a neuron that had T-type pre-inactivated
        let mut n2 = GranuleCell::new();
        n2.s = 0.1; // pre-inactivated
        let mut spikes_no_rebound = 0;
        for _ in 0..200 {
            spikes_no_rebound += n2.step(10.0);
        }

        assert!(
            spikes_early >= spikes_no_rebound,
            "De-inactivated T-type should facilitate firing: early={spikes_early} vs inactivated={spikes_no_rebound}"
        );
    }

    #[test]
    fn granule_negative_input_no_crash() {
        let mut n = GranuleCell::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite(), "Must stay finite with negative input");
        assert!(n.v >= -100.0, "Must be bounded");
    }

    #[test]
    fn granule_nan_input_stays_finite() {
        let mut n = GranuleCell::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite(), "NaN input must not corrupt state");
    }

    #[test]
    fn granule_extreme_input_bounded() {
        let mut n = GranuleCell::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0, "Extreme input must stay bounded");
    }

    #[test]
    fn granule_reset_clears_state() {
        let mut n = GranuleCell::new();
        for _ in 0..1000 {
            n.step(20.0);
        }
        n.reset();
        assert_eq!(n.v, -70.0);
        assert_eq!(n.s, 0.95);
        assert_eq!(n.refrac_count, 0.0);
    }

    #[test]
    fn granule_high_input_resistance() {
        // Small soma → large voltage response to small current
        let mut n = GranuleCell::new();
        let v_before = n.v;
        // Single step with moderate input
        n.step(5.0);
        let dv = n.v - v_before;
        assert!(dv > 0.5, "High Rin should give large voltage change, got dv={dv}");
    }

    #[test]
    fn granule_performance_10k_steps() {
        let start = std::time::Instant::now();
        let mut n = GranuleCell::new();
        for _ in 0..10_000 {
            std::hint::black_box(n.step(10.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "10k steps must complete in <50ms, took {}ms",
            elapsed.as_millis()
        );
    }
}
