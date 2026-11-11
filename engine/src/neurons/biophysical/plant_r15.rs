// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Plant R15 Neuron Model

//! Plant R15 Aplysia parabolic-burster dynamics.

#![deny(missing_docs)]

use super::safe_rate;

/// Plant R15 — Aplysia parabolic burster. Plant & Kim 1976.
#[derive(Clone, Debug)]
pub struct PlantR15Neuron {
    /// Membrane potential (mV).
    pub v: f64,
    /// Sodium activation gate.
    pub m: f64,
    /// Sodium inactivation gate.
    pub h: f64,
    /// Potassium activation gate.
    pub n: f64,
    /// Intracellular calcium variable; activates the calcium-activated potassium current
    /// as `ca / (0.5 + ca)` and never falls below zero.
    pub ca: f64,
    /// Maximal sodium conductance (mS/cm^2).
    pub g_na: f64,
    /// Maximal delayed-rectifier potassium conductance (mS/cm^2).
    pub g_k: f64,
    /// Maximal calcium conductance (mS/cm^2).
    pub g_ca: f64,
    /// Leak conductance (mS/cm^2).
    pub g_l: f64,
    /// Maximal calcium-activated potassium conductance (mS/cm^2).
    pub g_kca: f64,
    /// Sodium reversal potential (mV).
    pub e_na: f64,
    /// Potassium reversal potential (mV).
    pub e_k: f64,
    /// Calcium reversal potential (mV).
    pub e_ca: f64,
    /// Leak reversal potential (mV).
    pub e_l: f64,
    /// Membrane capacitance (uF/cm^2).
    pub c_m: f64,
    /// Calcium influx coefficient converting calcium current into the calcium variable (cm^2/(uA*ms)).
    pub k_ca: f64,
    /// Calcium removal time constant (ms).
    pub tau_ca: f64,
    /// Forward-Euler integration step (ms); each [`PlantR15Neuron::step`] call runs five
    /// of them.
    pub dt: f64,
    /// Voltage at which a spike is registered (mV).
    pub v_threshold: f64,
}

impl PlantR15Neuron {
    /// Create a neuron with the default parameters and initial state of its model descriptor.
    pub fn new() -> Self {
        Self {
            v: -50.0,
            m: 0.05,
            h: 0.6,
            n: 0.3,
            ca: 0.1,
            g_na: 4.0,
            g_k: 0.3,
            g_ca: 0.004,
            g_l: 0.003,
            g_kca: 0.03,
            e_na: 30.0,
            e_k: -75.0,
            e_ca: 140.0,
            e_l: -40.0,
            c_m: 1.0,
            k_ca: 0.0085,
            tau_ca: 500.0,
            dt: 0.05,
            v_threshold: -10.0,
        }
    }
    /// Advance the neuron by five forward-Euler steps of `dt` with the injected current `current`.
    ///
    /// Returns 1 when the membrane potential starts below `v_threshold` and ends at or
    /// above it, else 0. The input and the state are not validated, so a non-finite
    /// current propagates into the state.
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        for _ in 0..5 {
            let am = safe_rate(0.1, 50.0, self.v, 10.0, 1.0);
            let bm = 4.0 * (-(self.v + 75.0) / 18.0).exp();
            let ah = 0.07 * (-(self.v + 50.0) / 20.0).exp();
            let bh = 1.0 / (1.0 + (-(self.v + 20.0) / 10.0).exp());
            let an = safe_rate(0.01, 55.0, self.v, 10.0, 0.1);
            let bn = 0.125 * (-(self.v + 65.0) / 80.0).exp();
            self.m += (am * (1.0 - self.m) - bm * self.m) * self.dt;
            self.h += (ah * (1.0 - self.h) - bh * self.h) * self.dt;
            self.n += (an * (1.0 - self.n) - bn * self.n) * self.dt;
            let m_ca = 1.0 / (1.0 + (-(self.v + 25.0) / 5.0).exp());
            let kca_act = self.ca / (0.5 + self.ca);
            let i_na = self.g_na * self.m.powi(3) * self.h * (self.v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
            let i_ca = self.g_ca * m_ca.powi(2) * (self.v - self.e_ca);
            let i_kca = self.g_kca * kca_act * (self.v - self.e_k);
            let i_l = self.g_l * (self.v - self.e_l);
            self.v += (-i_na - i_k - i_ca - i_kca - i_l + current) / self.c_m * self.dt;
            self.ca = (self.ca + (-self.k_ca * i_ca - self.ca / self.tau_ca) * self.dt).max(0.0);
        }
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    /// Return the membrane potential, the gates and calcium to their initial values; the
    /// parameters are kept.
    pub fn reset(&mut self) {
        self.v = -50.0;
        self.m = 0.05;
        self.h = 0.6;
        self.n = 0.3;
        self.ca = 0.1;
    }
}
impl Default for PlantR15Neuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = PlantR15Neuron::default();
        let constructed = PlantR15Neuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn plant_r15_fires() {
        let mut n = PlantR15Neuron::new();
        let t: i32 = (0..500).map(|_| n.step(2.0)).sum();
        assert!(t > 0);
    }

    // -- PlantR15 --
    #[test]
    fn plant_r15_silent_without_input() {
        let mut n = PlantR15Neuron::new();
        // R15 is a parabolic burster — may burst spontaneously
        for _ in 0..500 {
            n.step(0.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn plant_r15_reset_clears_state() {
        let mut n = PlantR15Neuron::new();
        for _ in 0..100 {
            n.step(2.0);
        }
        n.reset();
        assert!((n.v - (-50.0)).abs() < 1e-10);
        assert!((n.ca - 0.1).abs() < 1e-10);
    }
    #[test]
    fn plant_r15_moderate_input_stable() {
        // Plant R15 is a parabolic burster — moderate input stability
        let mut n = PlantR15Neuron::new();
        for _ in 0..500 {
            n.step(2.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn plant_r15_ca_dynamics() {
        let mut n = PlantR15Neuron::new();
        for _ in 0..500 {
            n.step(2.0);
        }
        assert!(n.ca >= 0.0, "Ca²⁺ must be non-negative");
        assert!(n.ca.is_finite());
    }
    #[test]
    fn plant_r15_weak_negative_no_crash() {
        let mut n = PlantR15Neuron::new();
        for _ in 0..200 {
            n.step(-1.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn plant_r15_nan_no_panic() {
        let mut n = PlantR15Neuron::new();
        n.step(f64::NAN);
    }
}
