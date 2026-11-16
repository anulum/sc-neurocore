// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Brunel Balanced Network Model

#![deny(missing_docs)]

/// Brunel 2000 — balanced excitatory/inhibitory network mean-field.
///
/// Two coupled rate equations for excitatory (r_e) and inhibitory (r_i)
/// populations. The balance of E and I determines the dynamical regime:
/// - Asynchronous Irregular (AI): low rates, Poisson-like
/// - Synchronous Regular (SR): oscillatory, gamma-band
/// - Synchronous Irregular (SI): fast oscillations, irregular single units
///
/// tau_e * dr_e/dt = -r_e + phi(J_ee*r_e - J_ei*r_i + I_ext)
/// tau_i * dr_i/dt = -r_i + phi(J_ie*r_e - J_ii*r_i)
///
/// phi() is a threshold-linear transfer function.
///
/// Brunel, J Comput Neurosci 8:183, 2000.
#[derive(Clone, Debug)]
pub struct BrunelNetwork {
    /// Excitatory rate (Hz).
    pub r_e: f64,
    /// Inhibitory rate (Hz).
    pub r_i: f64,
    /// Excitatory time constant (ms).
    pub tau_e: f64,
    /// Inhibitory time constant (ms).
    pub tau_i: f64,
    /// E→E coupling.
    pub j_ee: f64,
    /// I→E coupling (inhibitory, positive value).
    pub j_ei: f64,
    /// E→I coupling.
    pub j_ie: f64,
    /// I→I coupling (inhibitory, positive value).
    pub j_ii: f64,
    /// Transfer function threshold.
    pub threshold: f64,
    /// Transfer function gain.
    pub gain_phi: f64,
    /// Euler step (ms).
    pub dt: f64,
    /// Excitatory rate whose upward crossing counts as an event (Hz).
    pub r_threshold: f64,
    /// Gain applied to the external input (dimensionless).
    pub gain: f64,
}

impl Default for BrunelNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl BrunelNetwork {
    /// Create a network at `r_e` = `r_i` = 0.1 with the default couplings and `dt` = 0.1 ms.
    pub fn new() -> Self {
        Self {
            r_e: 0.1,
            r_i: 0.1,
            tau_e: 20.0,
            tau_i: 10.0,
            j_ee: 0.2,
            j_ei: 0.8, // Strong I→E inhibition
            j_ie: 0.5,
            j_ii: 0.2,
            threshold: 0.0,
            gain_phi: 1.0,
            dt: 0.1,
            r_threshold: 1.0,
            gain: 1.0,
        }
    }

    /// Threshold-linear transfer function
    fn phi(&self, x: f64) -> f64 {
        if x > self.threshold {
            self.gain_phi * (x - self.threshold)
        } else {
            0.0
        }
    }

    /// Advance one forward-Euler step of both rate equations with the external input
    /// `gain * current`; return 1 when `r_e` crosses `r_threshold` upwards, else 0.
    ///
    /// Both drives use the old rates. The rates are then floored at 0 and capped at 200;
    /// a NaN rate, which both bounds keep, becomes 0.1. The input is not validated.
    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let r_e_prev = self.r_e;

        let drive_e = self.j_ee * self.r_e - self.j_ei * self.r_i + input;
        let drive_i = self.j_ie * self.r_e - self.j_ii * self.r_i;

        let dr_e = (-self.r_e + self.phi(drive_e)) / self.tau_e;
        let dr_i = (-self.r_i + self.phi(drive_i)) / self.tau_i;

        self.r_e += self.dt * dr_e;
        self.r_i += self.dt * dr_i;

        // Rates non-negative
        if self.r_e < 0.0 {
            self.r_e = 0.0;
        }
        if self.r_i < 0.0 {
            self.r_i = 0.0;
        }

        // Safety bounds
        if self.r_e > 200.0 {
            self.r_e = 200.0;
        }
        if self.r_i > 200.0 {
            self.r_i = 200.0;
        }
        if !self.r_e.is_finite() {
            self.r_e = 0.1;
        }
        if !self.r_i.is_finite() {
            self.r_i = 0.1;
        }

        // "Spike" when E rate crosses threshold
        if self.r_e >= self.r_threshold && r_e_prev < self.r_threshold {
            1
        } else {
            0
        }
    }

    /// Restore every field to `new()`, parameters included.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
#[path = "brunel_network_tests.rs"]
mod tests;
