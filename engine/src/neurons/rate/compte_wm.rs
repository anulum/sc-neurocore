// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Compte working-memory neuron model

/// Compte WM — NMDA-based working-memory neuron. Compte et al. 2000.
#[derive(Clone, Debug)]
pub struct CompteWMNeuron {
    pub v: f64,
    pub s_ampa: f64,
    pub s_nmda: f64,
    pub x_nmda: f64,
    pub s_gaba: f64,
    pub g_l: f64,
    pub g_ampa: f64,
    pub g_nmda: f64,
    pub g_gaba: f64,
    pub e_l: f64,
    pub e_exc: f64,
    pub e_inh: f64,
    pub c_m: f64,
    pub mg: f64,
    pub tau_ampa: f64,
    pub tau_nmda: f64,
    pub tau_x: f64,
    pub alpha_nmda: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub dt: f64,
}

impl CompteWMNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            s_ampa: 0.0,
            s_nmda: 0.0,
            x_nmda: 0.0,
            s_gaba: 0.0,
            g_l: 0.025,
            g_ampa: 0.005,
            g_nmda: 0.165,
            g_gaba: 0.013,
            e_l: -70.0,
            e_exc: 0.0,
            e_inh: -70.0,
            c_m: 0.5,
            mg: 1.0,
            tau_ampa: 2.0,
            tau_nmda: 100.0,
            tau_x: 2.0,
            alpha_nmda: 0.5,
            v_threshold: -50.0,
            v_reset: -55.0,
            dt: 0.1,
        }
    }
    pub fn step(&mut self, current: f64, spike_in: bool) -> i32 {
        if spike_in {
            self.s_ampa += 1.0;
            self.x_nmda += 1.0;
        }
        self.s_ampa *= (-self.dt / self.tau_ampa).exp();
        self.s_nmda += (-self.s_nmda / self.tau_nmda
            + self.alpha_nmda * self.x_nmda * (1.0 - self.s_nmda))
            * self.dt;
        self.x_nmda *= (-self.dt / self.tau_x).exp();
        self.s_gaba *= (-self.dt / 5.0).exp();
        let mg_block = 1.0 / (1.0 + self.mg / 3.57 * (-0.062 * self.v).exp());
        let i_l = self.g_l * (self.v - self.e_l);
        let i_ampa = self.g_ampa * self.s_ampa * (self.v - self.e_exc);
        let i_nmda = self.g_nmda * mg_block * self.s_nmda * (self.v - self.e_exc);
        let i_gaba = self.g_gaba * self.s_gaba * (self.v - self.e_inh);
        self.v += (-i_l - i_ampa - i_nmda - i_gaba + current) / self.c_m * self.dt;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.s_gaba += 1.0;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = self.e_l;
        self.s_ampa = 0.0;
        self.s_nmda = 0.0;
        self.x_nmda = 0.0;
        self.s_gaba = 0.0;
    }
}
impl Default for CompteWMNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compte_fires() {
        let mut n = CompteWMNeuron::new();
        let t: i32 = (0..500).map(|_| n.step(5.0, false)).sum();
        assert!(t > 0);
    }

    #[test]
    fn compte_reset() {
        let mut n = CompteWMNeuron::new();
        for _ in 0..100 {
            n.step(5.0, false);
        }
        n.reset();
        assert!((n.v - n.e_l).abs() < 1e-10);
    }

    #[test]
    fn compte_nan_no_panic() {
        CompteWMNeuron::new().step(f64::NAN, false);
    }
}
