// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — VIP interneuron model

/// VIP (vasoactive intestinal peptide) irregular-spiking interneuron.
///
/// Biophysics: Na+, K+, A-type K+ (Kv4, transient outward, causes
/// accommodation), leak. High input resistance, small soma.
/// Key properties: irregular/accommodating firing, disinhibitory
/// role (inhibits SST+ and PV+), bipolar morphology.
///
/// Based on Porter et al. 1998 / Bhatt et al. 2019 parameterisation.
#[derive(Clone, Debug)]
pub struct VIPNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub a: f64, // A-type K+ activation
    pub b: f64, // A-type K+ inactivation
    // Conductances
    pub g_na: f64,
    pub g_k: f64,
    pub g_a: f64,
    pub g_l: f64,
    // Reversal potentials
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl VIPNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.8,
            n: 0.1,
            a: 0.0,
            b: 0.9,
            g_na: 35.0, // Lower than PV+ (smaller soma)
            g_k: 6.0,
            g_a: 8.0,  // Strong A-current → accommodation
            g_l: 0.01, // High input resistance
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 0.5, // Small soma → low capacitance
            dt: 0.025,
            v_threshold: -20.0,
        }
    }

    /// Return `[dV, dh, dn, da, db]` of the five-state VIP system at one consistent
    /// state. All gates relax through sigmoidal steady states (no singularities).
    fn derivatives(&self, v: f64, h: f64, n: f64, a: f64, b: f64, current: f64) -> [f64; 5] {
        let m_inf = 1.0 / (1.0 + (-(v + 30.0) / 9.5).exp());
        let h_inf = 1.0 / (1.0 + ((v + 53.0) / 7.0).exp());
        let tau_h = 0.37 + 2.78 / (1.0 + ((v + 40.5) / 6.0).exp());
        let n_inf = 1.0 / (1.0 + (-(v + 30.0) / 10.0).exp());
        let tau_n = 0.37 + 1.85 / (1.0 + ((v + 27.0) / 15.0).exp());
        let a_inf = 1.0 / (1.0 + (-(v + 50.0) / 20.0).exp());
        let b_inf = 1.0 / (1.0 + ((v + 78.0) / 6.0).exp());
        let dh = (h_inf - h) / tau_h;
        let dn = (n_inf - n) / tau_n;
        let da = (a_inf - a) / 5.0;
        let db = (b_inf - b) / 50.0;
        let i_na = self.g_na * m_inf * m_inf * m_inf * h * (v - self.e_na);
        let i_k = self.g_k * n * n * n * n * (v - self.e_k);
        let i_a = self.g_a * a * a * a * b * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        let dv = (-i_na - i_k - i_a - i_l + current) / self.c_m;
        [dv, dh, dn, da, db]
    }

    /// Return one classical RK4 increment of `[V, h, n, a, b]`, holding `current`
    /// constant across the four stages.
    fn rk4_substep(&self, s: [f64; 5], current: f64) -> [f64; 5] {
        let dt = self.dt;
        let k1 = self.derivatives(s[0], s[1], s[2], s[3], s[4], current);
        let k2 = self.derivatives(
            s[0] + 0.5 * dt * k1[0],
            s[1] + 0.5 * dt * k1[1],
            s[2] + 0.5 * dt * k1[2],
            s[3] + 0.5 * dt * k1[3],
            s[4] + 0.5 * dt * k1[4],
            current,
        );
        let k3 = self.derivatives(
            s[0] + 0.5 * dt * k2[0],
            s[1] + 0.5 * dt * k2[1],
            s[2] + 0.5 * dt * k2[2],
            s[3] + 0.5 * dt * k2[3],
            s[4] + 0.5 * dt * k2[4],
            current,
        );
        let k4 = self.derivatives(
            s[0] + dt * k3[0],
            s[1] + dt * k3[1],
            s[2] + dt * k3[2],
            s[3] + dt * k3[3],
            s[4] + dt * k3[4],
            current,
        );
        let mut out = [0.0_f64; 5];
        for i in 0..5 {
            out[i] = s[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }
        out
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let mut s = [self.v, self.h, self.n, self.a, self.b];
        for _ in 0..4 {
            s = self.rk4_substep(s, current);
        }
        self.v = s[0];
        self.h = s[1];
        self.n = s[2];
        self.a = s[3];
        self.b = s[4];
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = -65.0;
        self.h = 0.8;
        self.n = 0.1;
        self.a = 0.0;
        self.b = 0.9;
    }
}

impl Default for VIPNeuron {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Chandelier Cell (Axo-Axonic)
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vip_fires_with_input() {
        let mut n = VIPNeuron::new();
        let spikes: i32 = (0..10000).map(|_| n.step(2.0)).sum();
        assert!(spikes > 0, "VIP must fire with sustained input");
    }

    #[test]
    fn vip_no_fire_without_input() {
        let mut n = VIPNeuron::new();
        let spikes: i32 = (0..5000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn vip_accommodation() {
        // A-current causes transient accommodation at spike onset.
        // Compare fresh neuron's first 100 steps vs steady-state.
        let mut n = VIPNeuron::new();
        // First 500 steps: A-current b gate is high → strong IA → suppresses early spikes
        let onset: i32 = (0..500).map(|_| n.step(3.0)).sum();
        // Skip 5000 steps to reach steady state
        for _ in 0..5000 {
            n.step(3.0);
        }
        // Next 500 steps at steady state
        let steady: i32 = (0..500).map(|_| n.step(3.0)).sum();
        // At steady state, b has dropped, IA is weaker → fires at least as much
        assert!(
            steady >= onset,
            "VIP steady-state ({steady}) should fire >= onset ({onset})"
        );
    }

    #[test]
    fn vip_reset_roundtrip() {
        let mut n = VIPNeuron::new();
        for _ in 0..5000 {
            n.step(3.0);
        }
        n.reset();
        let mut fresh = VIPNeuron::new();
        let r1: i32 = (0..2000).map(|_| n.step(3.0)).sum();
        let r2: i32 = (0..2000).map(|_| fresh.step(3.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn vip_voltage_bounded() {
        let mut n = VIPNeuron::new();
        for _ in 0..20000 {
            n.step(5.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    #[ignore = "wall-clock performance smoke; use Criterion benches for timing evidence"]
    fn vip_performance_10k_steps() {
        let mut n = VIPNeuron::new();
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            n.step(3.0);
        }
        assert!(start.elapsed().as_millis() < 100);
    }
}
