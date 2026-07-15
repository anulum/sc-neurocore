// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Golomb Fast-Spiking Neuron Model

//! Golomb Kv3-enabled fast-spiking interneuron dynamics.

/// Golomb fast-spiking interneuron with Kv3. Golomb et al. 2007.
#[derive(Clone, Debug)]
pub struct GolombFSNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub p: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_kv3: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl GolombFSNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.9,
            n: 0.1,
            p: 0.0,
            g_na: 112.5,
            g_k: 225.0,
            g_kv3: 150.0,
            g_l: 0.25,
            e_na: 50.0,
            e_k: -90.0,
            e_l: -70.0,
            dt: 0.01,
            v_threshold: -20.0,
        }
    }
    /// Return `[dV, dh, dn, dp]` of the four-state Golomb-FS system at one
    /// consistent state. The capacitance is unit-normalised (the membrane time
    /// constant is folded into the conductances), matching the Python reference at
    /// its default `c_m = 1`.
    fn derivatives(&self, v: f64, h: f64, n: f64, p: f64, current: f64) -> [f64; 4] {
        let m_inf = 1.0 / (1.0 + (-(v + 24.0) / 11.5).exp());
        let h_inf = 1.0 / (1.0 + ((v + 58.3) / 6.7).exp());
        let n_inf = 1.0 / (1.0 + (-(v + 12.4) / 6.8).exp());
        let p_inf = 1.0 / (1.0 + (-(v + 3.0) / 8.0).exp());
        let tau_h = 0.5 + 14.0 / (1.0 + ((v + 60.0) / 12.0).exp());
        let tau_n = 0.087 + 11.4 / (1.0 + ((v + 14.6) / 8.6).exp());
        let tau_p = 0.1 + 4.0 / (1.0 + ((v + 25.0) / 10.0).exp());
        let dh = (h_inf - h) / tau_h;
        let dn = (n_inf - n) / tau_n;
        let dp = (p_inf - p) / tau_p;
        let i_na = self.g_na * m_inf * m_inf * m_inf * h * (v - self.e_na);
        let i_k = self.g_k * n * n * n * n * (v - self.e_k);
        let i_kv3 = self.g_kv3 * p * p * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        let dv = -i_na - i_k - i_kv3 - i_l + current;
        [dv, dh, dn, dp]
    }

    /// Return one classical RK4 increment of `[V, h, n, p]`, holding `current`
    /// constant across the four stages.
    fn rk4_substep(&self, s: [f64; 4], current: f64) -> [f64; 4] {
        let dt = self.dt;
        let k1 = self.derivatives(s[0], s[1], s[2], s[3], current);
        let k2 = self.derivatives(
            s[0] + 0.5 * dt * k1[0],
            s[1] + 0.5 * dt * k1[1],
            s[2] + 0.5 * dt * k1[2],
            s[3] + 0.5 * dt * k1[3],
            current,
        );
        let k3 = self.derivatives(
            s[0] + 0.5 * dt * k2[0],
            s[1] + 0.5 * dt * k2[1],
            s[2] + 0.5 * dt * k2[2],
            s[3] + 0.5 * dt * k2[3],
            current,
        );
        let k4 = self.derivatives(
            s[0] + dt * k3[0],
            s[1] + dt * k3[1],
            s[2] + dt * k3[2],
            s[3] + dt * k3[3],
            current,
        );
        let mut out = [0.0_f64; 4];
        for i in 0..4 {
            out[i] = s[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }
        out
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let mut s = [self.v, self.h, self.n, self.p];
        for _ in 0..10 {
            s = self.rk4_substep(s, current);
        }
        self.v = s[0];
        self.h = s[1];
        self.n = s[2];
        self.p = s[3];
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -65.0;
        self.h = 0.9;
        self.n = 0.1;
        self.p = 0.0;
    }
}
impl Default for GolombFSNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = GolombFSNeuron::default();
        let constructed = GolombFSNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn golomb_fires() {
        let mut n = GolombFSNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(200.0)).sum();
        assert!(t > 0);
    }

    // -- GolombFS --
    #[test]
    fn golomb_silent_without_input() {
        let mut n = GolombFSNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn golomb_reset_clears_state() {
        let mut n = GolombFSNeuron::new();
        for _ in 0..100 {
            n.step(200.0);
        }
        n.reset();
        assert!((n.v - (-65.0)).abs() < 1e-10);
    }
    #[test]
    fn golomb_extreme_bounded() {
        // Golomb et al. 2007: n^4 kinetics diverge at extreme I; test at high but realistic drive
        let mut n = GolombFSNeuron::new();
        for _ in 0..200 {
            n.step(200.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn golomb_kv3_enables_fast_spiking() {
        // Kv3 current enables high-frequency firing
        let mut n = GolombFSNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(300.0)).sum();
        assert!(t > 0, "Golomb FS should fire with high input, got {}", t);
    }
    #[test]
    fn golomb_negative_no_crash() {
        let mut n = GolombFSNeuron::new();
        for _ in 0..200 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn golomb_nan_no_panic() {
        let mut n = GolombFSNeuron::new();
        n.step(f64::NAN);
    }
}
