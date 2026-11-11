// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Bertram et al. 2000 four-state phantom burster

//! Source equations 1–10 with the authors' BJ_00 parameter set.

#![deny(missing_docs)]

#[path = "sc_three_state_phantom.rs"]
mod sc_three_state_phantom;
pub use sc_three_state_phantom::SCThreeStatePhantomBurster;

/// Four-state phantom burster of Bertram et al. (2000), source equations 1-10 with the
/// authors' BJ_00 parameter set.
///
/// A calcium current with instantaneous activation, a fast potassium gate `n`, two slow
/// potassium-gated currents (`s1`, `s2`) and a leak act on one membrane potential.
/// Conductances are in pS and the capacitance in fF, so the injected current is in fA.
#[derive(Clone, Debug)]
pub struct BertramPhantomBurster {
    /// Membrane potential (mV).
    pub v: f64,
    /// Fast potassium activation gate; its rate is scaled by `lambda_n`.
    pub n: f64,
    /// First slow potassium gate; relaxes with `tau_s1`.
    pub s1: f64,
    /// Second slow potassium gate; relaxes with `tau_s2`.
    pub s2: f64,
    /// Fast potassium-gate rate multiplier (dimensionless).
    pub lambda_n: f64,
    /// Maximal calcium conductance (pS).
    pub g_ca: f64,
    /// Maximal potassium conductance (pS).
    pub g_k: f64,
    /// Maximal conductance of the first slow current (pS).
    pub g_s1: f64,
    /// Maximal conductance of the second slow current (pS).
    pub g_s2: f64,
    /// Leak conductance (pS).
    pub g_l: f64,
    /// Calcium reversal potential (mV).
    pub e_ca: f64,
    /// Potassium reversal potential (mV).
    pub e_k: f64,
    /// Leak reversal potential (mV).
    pub e_l: f64,
    /// Membrane capacitance (fF).
    pub c_m: f64,
    /// Half-activation voltage of the calcium current (mV).
    pub v_m: f64,
    /// Slope of the calcium activation function (mV).
    pub s_m: f64,
    /// Half-activation voltage of the potassium current (mV).
    pub v_n: f64,
    /// Slope of the potassium activation function (mV).
    pub s_n: f64,
    /// Half-activation voltage of the first slow variable (mV).
    pub v_s1: f64,
    /// Slope of the first slow-variable activation (mV).
    pub s_s1: f64,
    /// Half-activation voltage of the second slow variable (mV).
    pub v_s2: f64,
    /// Slope of the second slow-variable activation (mV).
    pub s_s2: f64,
    /// Maximum voltage-dependent potassium-gate time constant (ms).
    pub tau_n_bar: f64,
    /// Time constant of the first slow variable (ms).
    pub tau_s1: f64,
    /// Time constant of the second slow variable (ms).
    pub tau_s2: f64,
    /// Runge-Kutta step (ms); one per [`BertramPhantomBurster::step`] call.
    pub dt: f64,
    /// Voltage at which a spike is registered (mV).
    pub v_threshold: f64,
}

impl BertramPhantomBurster {
    /// Create a burster with the default parameters and initial state of its model descriptor.
    pub fn new() -> Self {
        Self {
            v: -43.0,
            n: 0.03,
            s1: 0.1,
            s2: 0.434,
            lambda_n: 1.1,
            g_ca: 280.0,
            g_k: 1300.0,
            g_s1: 20.0,
            g_s2: 32.0,
            g_l: 25.0,
            e_ca: 100.0,
            e_k: -80.0,
            e_l: -40.0,
            c_m: 4524.0,
            v_m: -22.0,
            s_m: 7.5,
            v_n: -9.0,
            s_n: 10.0,
            v_s1: -40.0,
            s_s1: 0.5,
            v_s2: -42.0,
            s_s2: 0.4,
            tau_n_bar: 9.09,
            tau_s1: 1000.0,
            tau_s2: 120_000.0,
            dt: 0.5,
            v_threshold: -20.0,
        }
    }

    fn boltz(v: f64, midpoint: f64, slope: f64) -> f64 {
        1.0 / (1.0 + ((midpoint - v) / slope).exp())
    }

    fn derivatives(&self, state: [f64; 4], current: f64) -> [f64; 4] {
        let [v, n, s1, s2] = state;
        let m_inf = Self::boltz(v, self.v_m, self.s_m);
        let n_inf = Self::boltz(v, self.v_n, self.s_n);
        let s1_inf = Self::boltz(v, self.v_s1, self.s_s1);
        let s2_inf = Self::boltz(v, self.v_s2, self.s_s2);
        let tau_n = self.tau_n_bar / (1.0 + ((v - self.v_n) / self.s_n).exp());
        let i_ca = self.g_ca * m_inf * (v - self.e_ca);
        let i_k = self.g_k * n * (v - self.e_k);
        let i_s1 = self.g_s1 * s1 * (v - self.e_k);
        let i_s2 = self.g_s2 * s2 * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        [
            (-i_ca - i_k - i_s1 - i_s2 - i_l + current) / self.c_m,
            self.lambda_n * (n_inf - n) / tau_n,
            (s1_inf - s1) / self.tau_s1,
            (s2_inf - s2) / self.tau_s2,
        ]
    }

    fn shifted(state: [f64; 4], derivative: [f64; 4], scale: f64) -> [f64; 4] {
        std::array::from_fn(|index| state[index] + scale * derivative[index])
    }

    /// Advance the burster by one classical Runge-Kutta step of `dt` with the injected
    /// current `current` (fA).
    ///
    /// Returns 1 when the membrane potential starts below `v_threshold` and ends at or
    /// above it, else 0. The input and the state are not validated, so a non-finite
    /// current propagates into the state.
    pub fn step(&mut self, current: f64) -> i32 {
        let previous_v = self.v;
        let state = [self.v, self.n, self.s1, self.s2];
        let k1 = self.derivatives(state, current);
        let k2 = self.derivatives(Self::shifted(state, k1, 0.5 * self.dt), current);
        let k3 = self.derivatives(Self::shifted(state, k2, 0.5 * self.dt), current);
        let k4 = self.derivatives(Self::shifted(state, k3, self.dt), current);
        let next: [f64; 4] = std::array::from_fn(|index| {
            state[index]
                + self.dt * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]) / 6.0
        });
        self.v = next[0];
        self.n = next[1];
        self.s1 = next[2];
        self.s2 = next[3];
        i32::from(self.v >= self.v_threshold && previous_v < self.v_threshold)
    }

    /// Return the membrane potential and the gates to their initial values; the parameters
    /// are kept.
    pub fn reset(&mut self) {
        self.v = -43.0;
        self.n = 0.03;
        self.s1 = 0.1;
        self.s2 = 0.434;
    }
}

impl Default for BertramPhantomBurster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_defaults_and_dynamic_n() {
        let mut model = BertramPhantomBurster::new();
        let previous_n = model.n;
        model.step(0.0);
        assert_ne!(model.n, previous_n);
    }

    #[test]
    fn reset_restores_source_state() {
        let mut model = BertramPhantomBurster::new();
        model.step(200.0);
        model.reset();
        assert_eq!(
            [model.v, model.n, model.s1, model.s2],
            [-43.0, 0.03, 0.1, 0.434]
        );
    }
}
