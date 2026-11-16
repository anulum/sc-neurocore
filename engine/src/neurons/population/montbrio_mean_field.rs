// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Montbrió-Pazó-Roxin Mean-Field Model

#![deny(missing_docs)]

/// Montbrio-Pazo-Roxin 2015 — exact mean-field of QIF neuron population.
///
/// Reduces an infinite population of quadratic integrate-and-fire (QIF)
/// neurons to 2 ODEs for the population firing rate r and mean membrane
/// potential v. This is mathematically exact (not an approximation) for
/// Lorentzian-distributed heterogeneity.
///
/// dr/dt = (delta / (pi * tau^2)) + (2 * r * v / tau)
/// dv/dt = (v^2 + eta + I + J * tau * r - (pi * tau * r)^2) / tau
///
/// where delta = heterogeneity width, eta = mean excitability,
/// tau = membrane time constant, J = synaptic coupling, I = external input.
///
/// Spike emitted when r exceeds a threshold (proxy for population burst).
///
/// Montbrio, Pazo & Roxin, Phys Rev X 5:021028, 2015.
#[derive(Clone, Debug)]
pub struct MontbrioMeanField {
    /// Population firing rate, in units of 1/`tau`, kept within `[0, 100]`.
    pub r: f64,
    /// Mean membrane potential.
    pub v: f64,
    /// Heterogeneity width (Lorentzian).
    pub delta: f64,
    /// Mean excitability.
    pub eta: f64,
    /// Membrane time constant (ms).
    pub tau: f64,
    /// Synaptic coupling strength.
    pub j: f64,
    /// Euler step, in the time unit of `tau`.
    pub dt: f64,
    /// Rate whose upward crossing counts as a population burst.
    pub r_threshold: f64,
    /// Gain applied to the external input (dimensionless).
    pub gain: f64,
}

impl Default for MontbrioMeanField {
    fn default() -> Self {
        Self::new()
    }
}

impl MontbrioMeanField {
    /// Create a population at `r` = 0.01 and `v` = -2 with the default parameters and
    /// `dt` = 0.01.
    pub fn new() -> Self {
        Self {
            r: 0.01,
            v: -2.0,
            delta: 1.0,
            eta: -5.0, // Below threshold for spontaneous activity
            tau: 1.0,
            j: 15.0,  // Excitatory coupling
            dt: 0.01, // Small dt for stability
            r_threshold: 0.5,
            gain: 1.0,
        }
    }

    /// Advance one forward-Euler step of both equations with the input `gain * current`;
    /// return 1 when `r` crosses `r_threshold` upwards, else 0.
    ///
    /// `r` is then clamped to `[0, 100]` and `v` to `[-50, 50]`, and a non-finite value is
    /// replaced by its initial value. The input is not validated.
    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let r_prev = self.r;

        let pi = std::f64::consts::PI;
        let tau = self.tau;

        // MPR equations with synaptic coupling (j * r adds recurrence)
        let dr = (self.delta / (pi * tau * tau)) + (2.0 * self.r * self.v / tau);
        let dv = (self.v * self.v + self.eta + input + self.j * tau * self.r
            - (pi * tau * self.r).powi(2))
            / tau;

        self.r += self.dt * dr;
        self.v += self.dt * dv;

        // Safety bounds
        self.r = self.r.clamp(0.0, 100.0);
        self.v = self.v.clamp(-50.0, 50.0);
        if !self.r.is_finite() {
            self.r = 0.01;
        }
        if !self.v.is_finite() {
            self.v = -2.0;
        }

        // "Spike" = population burst: r crosses threshold
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
#[path = "montbrio_mean_field_tests.rs"]
mod tests;
