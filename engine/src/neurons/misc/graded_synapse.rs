// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Graded Synapse Neuron Model

//! Non-spiking interneuron with graded transmitter release.

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

    #[test]
    fn graded_default_matches_constructor() {
        let default = GradedSynapseNeuron::default();
        let constructed = GradedSynapseNeuron::new();
        assert_eq!(default.v, constructed.v);
        assert_eq!(default.v_threshold, constructed.v_threshold);
        assert_eq!(default.dt, constructed.dt);
    }
}
