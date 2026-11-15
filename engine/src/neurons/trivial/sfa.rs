// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Spike-Frequency Adaptation Neuron

#![deny(missing_docs)]

/// Spike-Frequency Adaptation LIF. Benda & Herz 2003.
#[derive(Clone, Debug)]
pub struct SFANeuron {
    /// Membrane potential (mV).
    pub v: f64,
    /// Adaptation conductance relative to the leak (dimensionless); it decays with
    /// `tau_sfa` and gains `delta_g` at each spike.
    pub g_sfa: f64,
    /// Resting / leak reversal potential (mV).
    pub v_rest: f64,
    /// Membrane potential reset to after a spike (mV).
    pub v_reset: f64,
    /// Voltage at which a spike is registered (mV).
    pub v_threshold: f64,
    /// Membrane time constant (ms).
    pub tau_m: f64,
    /// Adaptation time constant of the spike-frequency-adaptation conductance (ms).
    pub tau_sfa: f64,
    /// Normalised adaptation-conductance increment added at each spike (dimensionless).
    pub delta_g: f64,
    /// Potassium (adaptation) reversal potential (mV).
    pub e_k: f64,
    /// Normalised membrane resistance (R=1 in the project's normalised-current convention) (dimensionless).
    pub resistance: f64,
    /// Integration step (ms).
    pub dt: f64,
}

impl SFANeuron {
    /// Create a neuron at rest with the descriptor defaults and `dt` = 1 ms.
    pub fn new() -> Self {
        Self {
            v: -70.0,
            g_sfa: 0.0,
            v_rest: -70.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau_m: 10.0,
            tau_sfa: 200.0,
            delta_g: 0.5,
            e_k: -80.0,
            resistance: 1.0,
            dt: 1.0,
        }
    }

    /// Advance one step with the input `current` and return 1 on a spike, else 0.
    ///
    /// `v` takes a forward-Euler step of
    /// `tau_m dv/dt = -(v - v_rest) - g_sfa * (v - e_k) + resistance * current`; then
    /// `g_sfa` decays by `exp(-dt / tau_sfa)`. Reaching `v_threshold` sets `v` to `v_reset`
    /// and adds `delta_g` to `g_sfa`. The input is not validated.
    pub fn step(&mut self, current: f64) -> i32 {
        self.v += (-(self.v - self.v_rest) - self.g_sfa * (self.v - self.e_k)
            + self.resistance * current)
            / self.tau_m
            * self.dt;
        self.g_sfa *= (-self.dt / self.tau_sfa).exp();
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.g_sfa += self.delta_g;
            1
        } else {
            0
        }
    }

    /// Return `v` to `v_rest` and clear `g_sfa`; the parameters are kept.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.g_sfa = 0.0;
    }
}

impl Default for SFANeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sfa_fires_then_adapts() {
        let mut n = SFANeuron::new();
        let first: i32 = (0..100).map(|_| n.step(30.0)).sum();
        let second: i32 = (0..100).map(|_| n.step(30.0)).sum();
        assert!(first > 0);
        assert!(second <= first + 2);
    }
    #[test]
    fn sfa_silent_without_input() {
        let mut n = SFANeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn sfa_reset_clears_state() {
        let mut n = SFANeuron::new();
        for _ in 0..100 {
            n.step(30.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
        assert!((n.g_sfa - 0.0).abs() < 1e-10);
    }
    #[test]
    fn sfa_bounded() {
        let mut n = SFANeuron::new();
        for _ in 0..1000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn sfa_nan_no_panic() {
        SFANeuron::new().step(f64::NAN);
    }
}
