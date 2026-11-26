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

    /// Advance one fourth-order Runge-Kutta step with the input `current` and return 1 on a
    /// spike, else 0.
    ///
    /// The derivatives are `dv = (-(v - v_rest) - g_sfa * (v - e_k) + resistance * current) /
    /// tau_m` and `dg_sfa = -g_sfa / tau_sfa`. The candidate `(v, g_sfa)` is the classic RK4
    /// combination of the four coupled slopes, evaluated in the same order of operations as
    /// the Python model, so the engine follows it bit for bit. When the candidate `v` reaches
    /// `v_threshold`, `v` is set to `v_reset` and `g_sfa` becomes the candidate plus `delta_g`;
    /// otherwise the candidate becomes the state. The input is not validated.
    pub fn step(&mut self, current: f64) -> i32 {
        let (v_candidate, g_candidate) = self.rk4_candidate(current);
        if v_candidate >= self.v_threshold {
            self.v = self.v_reset;
            self.g_sfa = g_candidate + self.delta_g;
            1
        } else {
            self.v = v_candidate;
            self.g_sfa = g_candidate;
            0
        }
    }

    fn derivatives(&self, v: f64, g_sfa: f64, current: f64) -> (f64, f64) {
        let dv =
            (-(v - self.v_rest) - g_sfa * (v - self.e_k) + self.resistance * current) / self.tau_m;
        let dg = -g_sfa / self.tau_sfa;
        (dv, dg)
    }

    fn rk4_candidate(&self, current: f64) -> (f64, f64) {
        let (k1v, k1g) = self.derivatives(self.v, self.g_sfa, current);
        let (k2v, k2g) = self.derivatives(
            self.v + 0.5 * self.dt * k1v,
            self.g_sfa + 0.5 * self.dt * k1g,
            current,
        );
        let (k3v, k3g) = self.derivatives(
            self.v + 0.5 * self.dt * k2v,
            self.g_sfa + 0.5 * self.dt * k2g,
            current,
        );
        let (k4v, k4g) =
            self.derivatives(self.v + self.dt * k3v, self.g_sfa + self.dt * k3g, current);
        (
            self.v + (self.dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
            self.g_sfa + (self.dt / 6.0) * (k1g + 2.0 * k2g + 2.0 * k3g + k4g),
        )
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
    fn sfa_rk4_matches_the_python_model_bit_for_bit() {
        // Python's SFANeuron after 50 steps of 30.0 from its defaults: one spike (step 11),
        // v = -52.00976080767626 and g_sfa = 0.41141732902842876.
        let mut n = SFANeuron::new();
        let spikes: Vec<i32> = (0..50).map(|_| n.step(30.0)).collect();
        assert_eq!(spikes.iter().sum::<i32>(), 1);
        assert_eq!(spikes[10], 1);
        assert_eq!(n.v, -52.00976080767626);
        assert_eq!(n.g_sfa, 0.41141732902842876);
    }

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
