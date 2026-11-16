// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — El Boustani Network Model

#![deny(missing_docs)]

/// El Boustani & Bhatt 2009 — E/I network with NMDA-mediated bistability.
///
/// Two-population (E/I) mean-field with separate fast (AMPA) and slow
/// (NMDA) excitatory components. NMDA provides the recurrent excitation
/// needed for working memory persistent activity (bistability).
///
/// tau_e * dr_e/dt = -r_e + phi(J_ampa*r_e + J_nmda*s + I - J_ei*r_i)
/// tau_i * dr_i/dt = -r_i + phi(J_ie*r_e - J_ii*r_i)
/// tau_s * ds/dt = -s + gamma * r_e * (1 - s)
///
/// El Boustani & Bhatt, J Comput Neurosci 26:313, 2009.
#[derive(Clone, Debug)]
pub struct ElBoustaniNetwork {
    /// Excitatory rate (Hz), kept within `[0, 200]`.
    pub r_e: f64,
    /// Inhibitory rate (Hz), kept within `[0, 200]`.
    pub r_i: f64,
    /// NMDA synaptic gating variable.
    pub s: f64,
    /// Excitatory time constant (ms).
    pub tau_e: f64,
    /// Inhibitory time constant (ms).
    pub tau_i: f64,
    /// NMDA decay (~100 ms).
    pub tau_s: f64,
    /// Fast E→E (AMPA).
    pub j_ampa: f64,
    /// Slow E→E (NMDA).
    pub j_nmda: f64,
    /// I→E coupling (inhibitory, positive value).
    pub j_ei: f64,
    /// E→I coupling.
    pub j_ie: f64,
    /// I→I coupling (inhibitory, positive value).
    pub j_ii: f64,
    /// NMDA saturation rate.
    pub gamma: f64,
    /// Threshold of the threshold-linear transfer function.
    pub threshold: f64,
    /// Slope of the transfer function above its threshold.
    pub gain_phi: f64,
    /// Euler step (ms).
    pub dt: f64,
    /// Excitatory rate whose upward crossing counts as an event (Hz).
    pub r_threshold: f64,
    /// Gain applied to the external input (dimensionless).
    pub gain: f64,
}

impl Default for ElBoustaniNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl ElBoustaniNetwork {
    /// Create a network at `r_e` = `r_i` = 0.1 and `s` = 0 with the default couplings and
    /// `dt` = 0.1 ms.
    pub fn new() -> Self {
        Self {
            r_e: 0.1,
            r_i: 0.1,
            s: 0.0,
            tau_e: 20.0,
            tau_i: 10.0,
            tau_s: 100.0,
            j_ampa: 0.1,
            j_nmda: 0.5,
            j_ei: 0.8,
            j_ie: 0.5,
            j_ii: 0.2,
            gamma: 0.641, // NMDA saturation parameter
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

    /// Advance one forward-Euler step with the external input `gain * current`; return 1
    /// when `r_e` crosses `r_threshold` upwards, else 0.
    ///
    /// `s` is updated first. The excitatory drive then uses the new `s` with the old rates.
    /// The rates are clamped to `[0, 200]` and `s` to `[0, 1]`, and a non-finite value is
    /// replaced by its initial value. The input is not validated.
    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let r_e_prev = self.r_e;

        // NMDA gating dynamics
        let ds = (-self.s + self.gamma * self.r_e * (1.0 - self.s)) / self.tau_s;
        self.s += self.dt * ds;

        // E and I rate dynamics
        let drive_e = self.j_ampa * self.r_e + self.j_nmda * self.s - self.j_ei * self.r_i + input;
        let drive_i = self.j_ie * self.r_e - self.j_ii * self.r_i;

        let dr_e = (-self.r_e + self.phi(drive_e)) / self.tau_e;
        let dr_i = (-self.r_i + self.phi(drive_i)) / self.tau_i;

        self.r_e += self.dt * dr_e;
        self.r_i += self.dt * dr_i;

        // Bounds
        self.r_e = self.r_e.clamp(0.0, 200.0);
        self.r_i = self.r_i.clamp(0.0, 200.0);
        self.s = self.s.clamp(0.0, 1.0);
        if !self.r_e.is_finite() {
            self.r_e = 0.1;
        }
        if !self.r_i.is_finite() {
            self.r_i = 0.1;
        }
        if !self.s.is_finite() {
            self.s = 0.0;
        }

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
#[path = "el_boustani_network_tests.rs"]
mod tests;
