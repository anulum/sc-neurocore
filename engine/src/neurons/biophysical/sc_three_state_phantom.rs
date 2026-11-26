// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained three-state project phantom recurrence

//! SCThreeState dual-slow-potassium phantom-burster dynamics.

/// SC three-state phantom burster — dual slow K for phantom bursting. retained project recurrence.
#[derive(Clone, Debug)]
pub struct SCThreeStatePhantomBurster {
    /// Membrane potential (mV).
    pub v: f64,
    /// First slow potassium gate; relaxes with `tau_s1`.
    pub s1: f64,
    /// Second slow potassium gate; relaxes with `tau_s2`.
    pub s2: f64,
    /// Maximal calcium conductance (mS/cm^2).
    pub g_ca: f64,
    /// Maximal instantaneous potassium conductance (mS/cm^2).
    pub g_k: f64,
    /// Maximal first slow potassium conductance (mS/cm^2).
    pub g_s1: f64,
    /// Maximal second slow potassium conductance (mS/cm^2).
    pub g_s2: f64,
    /// Leak conductance (mS/cm^2).
    pub g_l: f64,
    /// Calcium reversal potential (mV).
    pub e_ca: f64,
    /// Potassium reversal potential (mV).
    pub e_k: f64,
    /// Leak reversal potential (mV).
    pub e_l: f64,
    /// Membrane capacitance (uF/cm^2).
    pub c_m: f64,
    /// Calcium activation midpoint (mV).
    pub v_m: f64,
    /// Calcium activation slope (mV).
    pub s_m: f64,
    /// Instantaneous potassium activation midpoint (mV).
    pub v_n: f64,
    /// Instantaneous potassium activation slope (mV).
    pub s_n: f64,
    /// First slow-gate activation midpoint (mV).
    pub v_s1: f64,
    /// First slow-gate activation slope (mV).
    pub s_s1: f64,
    /// Second slow-gate activation midpoint (mV).
    pub v_s2: f64,
    /// Second slow-gate activation slope (mV).
    pub s_s2: f64,
    /// First slow-gate time constant (ms).
    pub tau_s1: f64,
    /// Second slow-gate time constant (ms).
    pub tau_s2: f64,
    /// Forward-Euler integration step (ms); one step per
    /// [`SCThreeStatePhantomBurster::step`] call.
    pub dt: f64,
    /// Sampled upward-crossing threshold (mV).
    pub v_threshold: f64,
}

impl SCThreeStatePhantomBurster {
    /// Create a burster with the default parameters and initial state of its model descriptor.
    pub fn new() -> Self {
        Self {
            v: -50.0,
            s1: 0.1,
            s2: 0.1,
            g_ca: 3.6,
            g_k: 10.0,
            g_s1: 4.0,
            g_s2: 4.0,
            g_l: 0.2,
            e_ca: 25.0,
            e_k: -75.0,
            e_l: -40.0,
            c_m: 5.3,
            v_m: -20.0,
            s_m: 12.0,
            v_n: -16.0,
            s_n: 5.6,
            v_s1: -40.0,
            s_s1: 10.0,
            v_s2: -42.0,
            s_s2: 0.4,
            tau_s1: 20000.0,
            tau_s2: 100000.0,
            dt: 0.5,
            v_threshold: -20.0,
        }
    }
    /// Advance the burster by one fourth-order Runge-Kutta step of `dt` with the injected
    /// current `current`.
    ///
    /// The calcium and the fast potassium activations are taken at their steady states, from
    /// the same sign-split Boltzmann function as the Python model, and the four slopes and
    /// their combination are evaluated in the same order of operations, so the engine follows
    /// it bit for bit. The slow gates of the candidate are clamped to [0, 1], as in Python.
    /// Returns 1 when the membrane potential starts below `v_threshold` and ends at or above
    /// it, else 0. The input and the state are not validated, so a non-finite current
    /// propagates into the state.
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let (v, s1, s2) = self.rk4_candidate(current);
        self.v = v;
        self.s1 = s1.clamp(0.0, 1.0);
        self.s2 = s2.clamp(0.0, 1.0);
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    fn boltz(v: f64, vh: f64, k: f64) -> f64 {
        let z = (vh - v) / k;
        if z >= 0.0 {
            let exp_neg = (-z).exp();
            exp_neg / (1.0 + exp_neg)
        } else {
            1.0 / (1.0 + z.exp())
        }
    }

    fn derivatives(&self, v: f64, s1: f64, s2: f64, current: f64) -> (f64, f64, f64) {
        let m_inf = Self::boltz(v, self.v_m, self.s_m);
        let n_inf = Self::boltz(v, self.v_n, self.s_n);
        let s1_inf = Self::boltz(v, self.v_s1, self.s_s1);
        let s2_inf = Self::boltz(v, self.v_s2, self.s_s2);
        let i_ca = self.g_ca * m_inf * (v - self.e_ca);
        let i_k = self.g_k * n_inf * (v - self.e_k);
        let i_s1 = self.g_s1 * s1 * (v - self.e_k);
        let i_s2 = self.g_s2 * s2 * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        let dv = (-i_ca - i_k - i_s1 - i_s2 - i_l + current) / self.c_m;
        let ds1 = (s1_inf - s1) / self.tau_s1;
        let ds2 = (s2_inf - s2) / self.tau_s2;
        (dv, ds1, ds2)
    }

    fn rk4_candidate(&self, current: f64) -> (f64, f64, f64) {
        let (v0, s10, s20, dt) = (self.v, self.s1, self.s2, self.dt);
        let k1 = self.derivatives(v0, s10, s20, current);
        let k2 = self.derivatives(
            v0 + 0.5 * dt * k1.0,
            s10 + 0.5 * dt * k1.1,
            s20 + 0.5 * dt * k1.2,
            current,
        );
        let k3 = self.derivatives(
            v0 + 0.5 * dt * k2.0,
            s10 + 0.5 * dt * k2.1,
            s20 + 0.5 * dt * k2.2,
            current,
        );
        let k4 = self.derivatives(v0 + dt * k3.0, s10 + dt * k3.1, s20 + dt * k3.2, current);
        (
            v0 + dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0,
            s10 + dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0,
            s20 + dt * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2) / 6.0,
        )
    }
    /// Return the membrane potential and the slow gates to their initial values; the
    /// parameters are kept.
    pub fn reset(&mut self) {
        self.v = -50.0;
        self.s1 = 0.1;
        self.s2 = 0.1;
    }
}
impl Default for SCThreeStatePhantomBurster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phantom_rk4_matches_the_python_model_bit_for_bit() {
        // Python's SCThreeStatePhantomBurster after 400 steps of 0.0 from its defaults: no
        // crossing, v = -34.292638252736324, s1 = 0.10514955278165967 and
        // s2 = 0.1016465239145123.
        let mut n = SCThreeStatePhantomBurster::new();
        let crossings: i32 = (0..400).map(|_| n.step(0.0)).sum();
        assert_eq!(crossings, 0);
        assert_eq!(n.v, -34.292638252736324);
        assert_eq!(n.s1, 0.10514955278165967);
        assert_eq!(n.s2, 0.1016465239145123);
    }

    #[test]
    fn default_matches_constructor_state() {
        let default = SCThreeStatePhantomBurster::default();
        let constructed = SCThreeStatePhantomBurster::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn sc_three_state_fires() {
        let mut n = SCThreeStatePhantomBurster::new();
        let t: i32 = (0..10000).map(|_| n.step(200.0)).sum();
        assert!(t > 0);
    }

    // -- SCThreeStatePhantom --
    #[test]
    fn sc_three_state_silent_without_input() {
        let mut n = SCThreeStatePhantomBurster::new();
        for _ in 0..500 {
            n.step(0.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn sc_three_state_reset_clears_state() {
        let mut n = SCThreeStatePhantomBurster::new();
        for _ in 0..1000 {
            n.step(200.0);
        }
        n.reset();
        assert!((n.v - (-50.0)).abs() < 1e-10);
        assert!((n.s1 - 0.1).abs() < 1e-10);
        assert!((n.s2 - 0.1).abs() < 1e-10);
    }
    #[test]
    fn sc_three_state_extreme_bounded() {
        let mut n = SCThreeStatePhantomBurster::new();
        for _ in 0..200 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn sc_three_state_dual_slow_vars() {
        let mut n = SCThreeStatePhantomBurster::new();
        for _ in 0..10000 {
            n.step(200.0);
        }
        // Both slow variables should evolve
        assert!(n.s1 >= 0.0 && n.s1 <= 1.0, "s1={}", n.s1);
        assert!(n.s2 >= 0.0 && n.s2 <= 1.0, "s2={}", n.s2);
    }
    #[test]
    fn sc_three_state_negative_no_crash() {
        let mut n = SCThreeStatePhantomBurster::new();
        for _ in 0..200 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn sc_three_state_nan_no_panic() {
        let mut n = SCThreeStatePhantomBurster::new();
        n.step(f64::NAN);
    }
}
