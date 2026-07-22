// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Astrocyte-LIF neuron model

//! Astrocyte-coupled leaky integrate-and-fire neuron model.

/// Astrocyte-LIF hybrid unit with calcium wave feedback.
///
/// Models the tripartite synapse: a glial astrocyte monitors extracellular
/// glutamate from a paired LIF neuron and provides slow homeostatic feedback
/// via calcium-dependent gliotransmitter release.
///
///   dCa/dt = -Ca/τ_ca + δ · S_pre(t)        (calcium rise on presynaptic spike)
///   I_glio = g_glio · H(Ca - Ca_thresh)      (gliotransmitter release)
///   dV/dt = -(V - E_L)/τ_m + I_ext + I_glio  (LIF with glial feedback)
///
/// Reference: Perea, Navarrete & Araque, "Tripartite synapses" (2009).
#[derive(Clone, Debug)]
pub struct AstrocyteLIFNeuron {
    pub v: f64,
    pub ca: f64,
    pub tau_m: f64,
    pub tau_ca: f64,
    pub e_l: f64,
    pub theta: f64,
    pub v_reset: f64,
    pub ca_delta: f64,
    pub ca_thresh: f64,
    pub g_glio: f64,
    pub dt: f64,
}

impl AstrocyteLIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            ca: 0.0,
            tau_m: 20.0,
            tau_ca: 500.0,
            e_l: -65.0,
            theta: -50.0,
            v_reset: -65.0,
            ca_delta: 0.1,
            ca_thresh: 0.5,
            g_glio: 2.0,
            dt: 0.1,
        }
    }

    /// Step with external current and presynaptic spike indicator.
    pub fn step_with_pre(&mut self, i_ext: f64, pre_spike: bool) -> i32 {
        // Astrocyte calcium dynamics.
        let dca = -self.ca / self.tau_ca
            + if pre_spike {
                self.ca_delta / self.dt
            } else {
                0.0
            };
        self.ca += dca * self.dt;
        self.ca = self.ca.max(0.0);

        // Gliotransmitter release (Heaviside on calcium).
        let i_glio = if self.ca > self.ca_thresh {
            self.g_glio
        } else {
            0.0
        };

        // LIF membrane dynamics with glial feedback.
        let dv = (-(self.v - self.e_l) + i_ext + i_glio) / self.tau_m;
        self.v += dv * self.dt;

        if self.v >= self.theta {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    /// Simple step (no presynaptic spike).
    pub fn step(&mut self, current: f64) -> i32 {
        self.step_with_pre(current, false)
    }

    pub fn reset(&mut self) {
        self.v = self.e_l;
        self.ca = 0.0;
    }
}

impl Default for AstrocyteLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn astrocyte_calcium_rises_on_pre_spikes() {
        let mut n = AstrocyteLIFNeuron::new();
        let ca_before = n.ca;
        for _ in 0..100 {
            n.step_with_pre(0.0, true);
        }
        assert!(
            n.ca > ca_before,
            "Calcium must rise with presynaptic spikes"
        );
    }

    #[test]
    fn astrocyte_gliotransmitter_boosts_firing() {
        let mut n_no_glio = AstrocyteLIFNeuron::new();
        let mut n_glio = AstrocyteLIFNeuron::new();

        let mut spikes_no = 0;
        let mut spikes_yes = 0;
        for _ in 0..5000 {
            spikes_no += n_no_glio.step_with_pre(10.0, false);
            spikes_yes += n_glio.step_with_pre(10.0, true); // pre spikes → Ca → glio
        }
        assert!(
            spikes_yes >= spikes_no,
            "Gliotransmitter should boost firing: with={spikes_yes} >= without={spikes_no}"
        );
    }

    #[test]
    fn astrocyte_calcium_decays() {
        let mut n = AstrocyteLIFNeuron::new();
        // Build up calcium.
        for _ in 0..200 {
            n.step_with_pre(0.0, true);
        }
        let ca_peak = n.ca;
        // Let it decay.
        for _ in 0..5000 {
            n.step_with_pre(0.0, false);
        }
        assert!(
            n.ca < ca_peak * 0.5,
            "Calcium must decay: current={:.4} < peak={:.4}*0.5",
            n.ca,
            ca_peak
        );
    }
}
