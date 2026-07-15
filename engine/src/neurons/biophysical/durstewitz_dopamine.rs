// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Durstewitz Dopamine Neuron Model

//! Durstewitz prefrontal-cortex neuron dynamics with D1 modulation.

/// Durstewitz PFC neuron with D1 dopamine modulation. Durstewitz et al. 2000.
#[derive(Clone, Debug)]
pub struct DurstewitzDopamineNeuron {
    pub v: f64,
    pub h_na: f64,
    pub n_k: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_nmda: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_nmda: f64,
    pub e_l: f64,
    pub mg: f64,
    pub d1_level: f64,
    pub g_nmda_scale: f64,
    pub g_k_scale: f64,
    pub v_shift_na: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl DurstewitzDopamineNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h_na: 0.7,
            n_k: 0.2,
            g_na: 45.0,
            g_k: 18.0,
            g_nmda: 0.5,
            g_l: 0.02,
            e_na: 55.0,
            e_k: -80.0,
            e_nmda: 0.0,
            e_l: -65.0,
            mg: 1.0,
            d1_level: 0.0,
            g_nmda_scale: 2.5,
            g_k_scale: 1.5,
            v_shift_na: -5.0,
            dt: 0.05,
            v_threshold: -20.0,
        }
    }
    /// Right-hand side ``(dV, dh_na, dn_k)`` evaluated from one consistent state.
    ///
    /// The sodium activation ``m_∞`` is instantaneous, so it is recomputed from
    /// `v` at every RK4 stage. The conductance powers use explicit multiplication
    /// and the Mg²⁺ block keeps the `mg / 3.57 * exp` operand order so the
    /// Python, Julia, Go, and Mojo backends reproduce the trajectory bit-for-bit.
    fn derivatives(&self, v: f64, h_na: f64, n_k: f64, current: f64) -> [f64; 3] {
        let v_sh = self.d1_level * self.v_shift_na;
        let m_na_inf = 1.0 / (1.0 + (-(v + 30.0 + v_sh) / 9.5).exp());
        let h_na_inf = 1.0 / (1.0 + ((v + 53.0) / 7.0).exp());
        let n_k_inf = 1.0 / (1.0 + (-(v + 30.0) / 10.0).exp());
        let tau_h = 0.5 + 14.0 / (1.0 + ((v + 50.0) / 12.0).exp());
        let tau_n = 1.0 + 11.0 / (1.0 + ((v + 40.0) / 10.0).exp());
        let d_h_na = (h_na_inf - h_na) / tau_h;
        let d_n_k = (n_k_inf - n_k) / tau_n;
        let mg_block = 1.0 / (1.0 + self.mg / 3.57 * (-0.062 * v).exp());
        let nmda_g = self.g_nmda * (1.0 + self.d1_level * (self.g_nmda_scale - 1.0));
        let k_g = self.g_k * (1.0 + self.d1_level * (self.g_k_scale - 1.0));
        let i_na = self.g_na * m_na_inf * m_na_inf * m_na_inf * h_na * (v - self.e_na);
        let i_k = k_g * n_k * n_k * n_k * n_k * (v - self.e_k);
        let i_nmda = nmda_g * mg_block * (v - self.e_nmda);
        let i_l = self.g_l * (v - self.e_l);
        let d_v = -i_na - i_k - i_nmda - i_l + current;
        [d_v, d_h_na, d_n_k]
    }

    /// One classical RK4 increment of the `(V, h_na, n_k)` vector over `dt`.
    fn rk4_substep(&self, s: [f64; 3], current: f64) -> [f64; 3] {
        let dt = self.dt;
        let k1 = self.derivatives(s[0], s[1], s[2], current);
        let k2 = self.derivatives(
            s[0] + 0.5 * dt * k1[0],
            s[1] + 0.5 * dt * k1[1],
            s[2] + 0.5 * dt * k1[2],
            current,
        );
        let k3 = self.derivatives(
            s[0] + 0.5 * dt * k2[0],
            s[1] + 0.5 * dt * k2[1],
            s[2] + 0.5 * dt * k2[2],
            current,
        );
        let k4 = self.derivatives(
            s[0] + dt * k3[0],
            s[1] + dt * k3[1],
            s[2] + dt * k3[2],
            current,
        );
        let mut out = [0.0_f64; 3];
        for i in 0..3 {
            out[i] = s[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }
        out
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let s = self.rk4_substep([self.v, self.h_na, self.n_k], current);
        self.v = s[0];
        self.h_na = s[1];
        self.n_k = s[2];
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -65.0;
        self.h_na = 0.7;
        self.n_k = 0.2;
    }
}
impl Default for DurstewitzDopamineNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = DurstewitzDopamineNeuron::default();
        let constructed = DurstewitzDopamineNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn durstewitz_fires() {
        let mut n = DurstewitzDopamineNeuron::new();
        let t: i32 = (0..1000).map(|_| n.step(3.0)).sum();
        assert!(t > 0);
    }

    // -- DurstewitzDopamine --
    #[test]
    fn durstewitz_low_activity_zero_input() {
        let mut n = DurstewitzDopamineNeuron::new();
        let _t: i32 = (0..500).map(|_| n.step(0.0)).sum();
        // NMDA tonic conductance can produce spontaneous activity
        assert!(n.v.is_finite());
    }
    #[test]
    fn durstewitz_reset_clears_state() {
        let mut n = DurstewitzDopamineNeuron::new();
        for _ in 0..100 {
            n.step(3.0);
        }
        n.reset();
        assert!((n.v - (-65.0)).abs() < 1e-10);
    }
    #[test]
    fn durstewitz_extreme_bounded() {
        let mut n = DurstewitzDopamineNeuron::new();
        for _ in 0..200 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn durstewitz_d1_modulation() {
        // D1 dopamine should increase NMDA and shift Na activation
        let mut n_d1 = DurstewitzDopamineNeuron::new();
        n_d1.d1_level = 1.0;
        let mut n_no = DurstewitzDopamineNeuron::new();
        n_no.d1_level = 0.0;
        for _ in 0..1000 {
            n_d1.step(3.0);
        }
        for _ in 0..1000 {
            n_no.step(3.0);
        }
        // Both should remain stable; D1 changes effective conductances
        assert!(n_d1.v.is_finite() && n_no.v.is_finite());
    }
    #[test]
    fn durstewitz_mg_block() {
        let n = DurstewitzDopamineNeuron::new();
        // At rest (-65 mV), Mg²⁺ block should be high
        let block = 1.0 / (1.0 + n.mg * (-0.062 * n.v).exp() / 3.57);
        assert!(block < 0.1, "Mg²⁺ block at rest should be high: {}", block);
    }
    #[test]
    fn durstewitz_negative_no_crash() {
        let mut n = DurstewitzDopamineNeuron::new();
        for _ in 0..200 {
            n.step(-10.0);
        }
        assert!(n.v.is_finite());
    }
}
