// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Tsodyks-Markram neuron model

#![deny(missing_docs)]

/// Tsodyks-Markram 1997 — LIF with short-term synaptic plasticity.
#[derive(Clone, Debug)]
pub struct TsodyksMarkramNeuron {
    /// Membrane potential (mV).
    pub v: f64,
    /// Available synaptic resources (dimensionless); they recover towards 1 with `tau_d`.
    pub x: f64,
    /// Utilisation (dimensionless); it relaxes towards `u_se` with `tau_f`.
    pub u: f64,
    /// Resting potential (mV).
    pub v_rest: f64,
    /// Reset potential (mV).
    pub v_reset: f64,
    /// Firing threshold (mV).
    pub v_threshold: f64,
    /// Membrane time constant (ms).
    pub tau_m: f64,
    /// Synaptic depression recovery time constant (ms).
    pub tau_d: f64,
    /// Synaptic facilitation time constant (ms).
    pub tau_f: f64,
    /// Utilisation of synaptic efficacy (release probability) (dimensionless).
    pub u_se: f64,
    /// Absolute synaptic efficacy (maximal synaptic weight) (dimensionless).
    pub a_se: f64,
    /// Normalised membrane resistance (dimensionless).
    pub r_m: f64,
    /// Step (ms).
    pub dt: f64,
}

impl TsodyksMarkramNeuron {
    /// Create a neuron at rest with full resources (`x` = 1), `u` = `u_se`, the default
    /// parameters and `dt` = 0.1 ms.
    pub fn new() -> Self {
        Self {
            v: -65.0,
            x: 1.0,
            u: 0.2,
            v_rest: -65.0,
            v_reset: -65.0,
            v_threshold: -50.0,
            tau_m: 20.0,
            tau_d: 200.0,
            tau_f: 600.0,
            u_se: 0.2,
            a_se: 50.0,
            r_m: 1.0,
            dt: 0.1,
        }
    }
    /// Advance one forward-Euler step and return 1 on a spike, else 0.
    ///
    /// `x` recovers and `u` relaxes first. On a presynaptic spike, `u` gains
    /// `u_se * (1 - u)`, the synaptic current is `a_se * u * x` for this step only, and `x`
    /// loses `u * x`. `v` then gains `(-(v - v_rest) + r_m * (i_syn + current)) / tau_m * dt`,
    /// and reaching `v_threshold` sets it to `v_reset`. Inputs are not validated.
    pub fn step(&mut self, current: f64, presynaptic_spike: bool) -> i32 {
        self.x += (1.0 - self.x) / self.tau_d * self.dt;
        self.u += (self.u_se - self.u) / self.tau_f * self.dt;
        let mut i_syn = 0.0;
        if presynaptic_spike {
            self.u += self.u_se * (1.0 - self.u);
            i_syn = self.a_se * self.u * self.x;
            self.x -= self.u * self.x;
        }
        self.v += (-(self.v - self.v_rest) + self.r_m * (i_syn + current)) / self.tau_m * self.dt;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }
    /// Return `v` to `v_rest`, `x` to 1 and `u` to `u_se`; the parameters are kept.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.x = 1.0;
        self.u = self.u_se;
    }
}
impl Default for TsodyksMarkramNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tm_fires() {
        let mut n = TsodyksMarkramNeuron::new();
        let t: i32 = (0..500).map(|_| n.step(50.0, false)).sum();
        assert!(t > 0);
    }

    #[test]
    fn tm_reset() {
        let mut n = TsodyksMarkramNeuron::new();
        for _ in 0..100 {
            n.step(50.0, false);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
        assert!((n.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn tm_bounded() {
        let mut n = TsodyksMarkramNeuron::new();
        for _ in 0..1000 {
            n.step(1e4, false);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn tm_nan_no_panic() {
        TsodyksMarkramNeuron::new().step(f64::NAN, false);
    }

    #[test]
    fn tm_stp_depression() {
        let mut n = TsodyksMarkramNeuron::new();
        for _ in 0..500 {
            n.step(50.0, true);
        }
        // With repeated presynaptic spikes, x (available fraction) should decrease
        assert!(
            n.x < 1.0,
            "STP depression: x should be < 1.0 after spikes: {}",
            n.x
        );
    }
}
