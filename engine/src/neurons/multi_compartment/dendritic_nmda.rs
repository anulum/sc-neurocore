// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dendritic NMDA neuron model

//! Dendritic NMDA neuron model.

/// Dendritic NMDA spike model.
///
/// Captures the non-linear voltage-dependent Mg²⁺ block of NMDA receptors
/// in dendritic branches. NMDA current has a sigmoidal voltage dependence:
///
///   I_NMDA = g_NMDA · B(V) · (V - E_NMDA)
///   B(V) = 1 / (1 + [Mg²⁺]/3.57 · exp(-0.062 · V))
///
/// This enables coincidence detection: the dendrite only passes current
/// when both presynaptic glutamate AND postsynaptic depolarisation are present.
///
/// Reference: Jahr & Stevens (1990), Schiller et al. (2000).
#[derive(Clone, Debug)]
pub struct DendriticNMDANeuron {
    pub v_soma: f64,
    pub v_dend: f64,
    pub g_nmda: f64,
    pub e_nmda: f64,
    pub mg_conc: f64,
    pub g_coupling: f64,
    pub tau_soma: f64,
    pub tau_dend: f64,
    pub theta: f64,
    pub dt: f64,
}

impl DendriticNMDANeuron {
    pub fn new() -> Self {
        Self {
            v_soma: -65.0,
            v_dend: -65.0,
            g_nmda: 1.5,
            e_nmda: 0.0,
            mg_conc: 1.0,
            g_coupling: 0.5,
            tau_soma: 20.0,
            tau_dend: 50.0,
            theta: -50.0,
            dt: 0.1,
        }
    }

    /// Mg²⁺ block factor (Jahr & Stevens 1990).
    fn mg_block(&self, v: f64) -> f64 {
        1.0 / (1.0 + (self.mg_conc / 3.57) * (-0.062 * v).exp())
    }

    fn valid(&self) -> bool {
        self.v_soma.is_finite()
            && self.v_dend.is_finite()
            && self.g_nmda.is_finite()
            && self.g_nmda >= 0.0
            && self.e_nmda.is_finite()
            && self.mg_conc.is_finite()
            && self.mg_conc >= 0.0
            && self.g_coupling.is_finite()
            && self.g_coupling >= 0.0
            && self.tau_soma.is_finite()
            && self.tau_soma > 0.0
            && self.tau_dend.is_finite()
            && self.tau_dend > 0.0
            && self.theta.is_finite()
            && self.dt.is_finite()
            && self.dt > 0.0
    }

    fn derivatives(&self, v_soma: f64, v_dend: f64, i_soma: f64, glutamate: f64) -> (f64, f64) {
        let b = self.mg_block(v_dend);
        let i_nmda = self.g_nmda * glutamate * b * (v_dend - self.e_nmda);
        let dv_soma =
            (-v_soma - 65.0 + i_soma + self.g_coupling * (v_dend - v_soma)) / self.tau_soma;
        let dv_dend =
            (-v_dend - 65.0 + i_nmda + self.g_coupling * (v_soma - v_dend)) / self.tau_dend;
        (dv_soma, dv_dend)
    }

    fn rk4_substep(&self, v_soma: f64, v_dend: f64, i_soma: f64, glutamate: f64) -> (f64, f64) {
        let dt = self.dt;
        let (k1s, k1d) = self.derivatives(v_soma, v_dend, i_soma, glutamate);
        let (k2s, k2d) = self.derivatives(
            v_soma + 0.5 * dt * k1s,
            v_dend + 0.5 * dt * k1d,
            i_soma,
            glutamate,
        );
        let (k3s, k3d) = self.derivatives(
            v_soma + 0.5 * dt * k2s,
            v_dend + 0.5 * dt * k2d,
            i_soma,
            glutamate,
        );
        let (k4s, k4d) = self.derivatives(v_soma + dt * k3s, v_dend + dt * k3d, i_soma, glutamate);
        (
            v_soma + dt * (k1s + 2.0 * k2s + 2.0 * k3s + k4s) / 6.0,
            v_dend + dt * (k1d + 2.0 * k2d + 2.0 * k3d + k4d) / 6.0,
        )
    }

    /// Step with somatic input and dendritic glutamate.
    pub fn step(&mut self, i_soma: f64, glutamate: f64) -> i32 {
        if !i_soma.is_finite() || !glutamate.is_finite() || glutamate < 0.0 || !self.valid() {
            return 0;
        }
        let (next_v_soma, next_v_dend) =
            self.rk4_substep(self.v_soma, self.v_dend, i_soma, glutamate);
        if !next_v_soma.is_finite() || !next_v_dend.is_finite() {
            return 0;
        }
        self.v_dend = next_v_dend;
        if next_v_soma >= self.theta {
            self.v_soma = -65.0;
            1
        } else {
            self.v_soma = next_v_soma;
            0
        }
    }

    pub fn reset(&mut self) {
        self.v_soma = -65.0;
        self.v_dend = -65.0;
    }
}

impl Default for DendriticNMDANeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nmda_coincidence_detection() {
        let mut n = DendriticNMDANeuron::new();
        // Only soma input — dendrite contributes little.
        let mut spikes_soma_only = 0;
        for _ in 0..2000 {
            spikes_soma_only += n.step(8.0, 0.0);
        }
        n.reset();
        // Soma + glutamate — NMDA amplifies.
        let mut spikes_both = 0;
        for _ in 0..2000 {
            spikes_both += n.step(8.0, 1.0);
        }
        // Coincidence: both inputs together should fire more.
        assert!(
            spikes_both >= spikes_soma_only,
            "NMDA coincidence: both={spikes_both} must >= soma_only={spikes_soma_only}"
        );
    }

    #[test]
    fn nmda_mg_block_voltage_dependent() {
        let n = DendriticNMDANeuron::new();
        let b_hyper = n.mg_block(-80.0);
        let b_depol = n.mg_block(-20.0);
        assert!(
            b_depol > b_hyper,
            "Mg block must relieve at depolarised potentials: B(-20)={b_depol:.3} > B(-80)={b_hyper:.3}"
        );
    }

    #[test]
    fn nmda_zero_glutamate_no_nmda_current() {
        let mut n = DendriticNMDANeuron::new();
        let spikes: i32 = (0..500).map(|_| n.step(0.0, 0.0)).sum();
        assert_eq!(spikes, 0, "No input → no spikes");
    }

    #[test]
    fn nmda_rk4_cross_backend_anchor() {
        let mut n = DendriticNMDANeuron::new();
        let spikes: i32 = (0..20_000).map(|_| n.step(50.0, 0.5)).sum();
        assert_eq!(spikes, 253);
        assert!(n.v_soma.is_finite());
        assert!(n.v_dend.is_finite());
    }

    #[test]
    fn nmda_invalid_input_preserves_state() {
        let mut n = DendriticNMDANeuron::new();
        for _ in 0..10 {
            let _ = n.step(50.0, 0.5);
        }
        let old = (n.v_soma, n.v_dend);
        assert_eq!(n.step(f64::INFINITY, 0.5), 0);
        assert_eq!((n.v_soma, n.v_dend), old);
        assert_eq!(n.step(50.0, -1.0), 0);
        assert_eq!((n.v_soma, n.v_dend), old);
    }

    #[test]
    fn nmda_invalid_configuration_preserves_state() {
        let mut n = DendriticNMDANeuron::new();
        for _ in 0..10 {
            let _ = n.step(50.0, 0.5);
        }
        let old = (n.v_soma, n.v_dend);
        n.tau_dend = 0.0;
        assert_eq!(n.step(50.0, 0.5), 0);
        assert_eq!((n.v_soma, n.v_dend), old);
    }
}
