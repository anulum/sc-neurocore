// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Tsodyks-Uziel-Markram Network Model

#![deny(missing_docs)]

/// TUM 2000 — mean-field network with short-term synaptic plasticity.
///
/// Population rate equation coupled with Tsodyks-Markram STP dynamics
/// (synaptic depression and facilitation). The effective synaptic weight
/// is u*x*J, where u = utilisation (facilitation) and x = available
/// resources (depression).
///
/// tau * dr/dt = -r + phi(u * x * J * r + I)
/// dx/dt = (1 - x) / tau_d - u * x * r      (depression)
/// du/dt = (U - u) / tau_f + U * (1 - u) * r (facilitation)
///
/// Tsodyks, Uziel & Markram, J Neurosci 20:RC50, 2000.
#[derive(Clone, Debug)]
pub struct TUMNetwork {
    /// Population rate.
    pub r: f64,
    /// Available synaptic resources \[0, 1\].
    pub x: f64,
    /// Release probability (facilitation) \[0, 1\].
    pub u: f64,
    /// Base synaptic strength.
    pub j: f64,
    /// Baseline release probability.
    pub u_base: f64,
    /// Rate time constant (ms).
    pub tau: f64,
    /// Depression recovery (ms).
    pub tau_d: f64,
    /// Facilitation decay (ms).
    pub tau_f: f64,
    /// Threshold of the threshold-linear transfer function.
    pub threshold: f64,
    /// Slope of the transfer function above its threshold.
    pub gain_phi: f64,
    /// Euler step (ms).
    pub dt: f64,
    /// Rate whose upward crossing counts as an event.
    pub r_threshold: f64,
    /// Gain applied to the external input (dimensionless).
    pub gain: f64,
}

impl Default for TUMNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl TUMNetwork {
    /// Create a network at `r` = 0.1 with full resources (`x` = 1), `u` = 0.2, the default
    /// parameters and `dt` = 0.1 ms.
    pub fn new() -> Self {
        Self {
            r: 0.1,
            x: 1.0, // Full resources
            u: 0.2, // Low initial utilisation
            j: 5.0,
            u_base: 0.2,
            tau: 10.0,
            tau_d: 200.0, // Slow depression recovery
            tau_f: 50.0,  // Faster facilitation decay
            threshold: 0.0,
            gain_phi: 1.0,
            dt: 0.1,
            r_threshold: 1.0,
            gain: 1.0,
        }
    }

    fn phi(&self, x: f64) -> f64 {
        if x > self.threshold {
            self.gain_phi * (x - self.threshold)
        } else {
            0.0
        }
    }

    /// Advance one forward-Euler step with the input `gain * current`; return 1 when `r`
    /// crosses `r_threshold` upwards, else 0.
    ///
    /// `x` and `u` are updated first, from the old rate. The rate then uses the effective
    /// coupling `u * x * j` with the new `u` and `x`. `r` is clamped to `[0, 200]` and `x`
    /// and `u` to `[0, 1]`, and a non-finite value is replaced by its initial value. The
    /// input is not validated.
    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let r_prev = self.r;

        // STP dynamics
        let dx = (1.0 - self.x) / self.tau_d - self.u * self.x * self.r;
        let du = (self.u_base - self.u) / self.tau_f + self.u_base * (1.0 - self.u) * self.r;

        self.x += self.dt * dx;
        self.u += self.dt * du;

        // Rate dynamics with STP-modulated coupling
        let effective_j = self.u * self.x * self.j;
        let dr = (-self.r + self.phi(effective_j * self.r + input)) / self.tau;
        self.r += self.dt * dr;

        // Bounds
        self.r = self.r.clamp(0.0, 200.0);
        self.x = self.x.clamp(0.0, 1.0);
        self.u = self.u.clamp(0.0, 1.0);
        if !self.r.is_finite() {
            self.r = 0.1;
        }
        if !self.x.is_finite() {
            self.x = 1.0;
        }
        if !self.u.is_finite() {
            self.u = 0.2;
        }

        if self.r >= self.r_threshold && r_prev < self.r_threshold {
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
#[path = "tum_network_tests.rs"]
mod tests;
