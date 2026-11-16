// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Floating-point Izhikevich neuron

#![deny(missing_docs)]

/// Izhikevich neuron (floating-point).
///
/// Standard model from IEEE TNN 14(6), 2003:
///   v' = 0.04*v² + 5*v + 140 - u + I
///   u' = a*(b*v - u)
///   if v >= 30: v ← c, u ← u + d
#[derive(Clone, Debug)]
pub struct Izhikevich {
    /// Membrane potential (mV).
    pub v: f64,
    /// Recovery variable.
    pub u: f64,
    /// Time scale of the recovery variable.
    pub a: f64,
    /// Sensitivity of the recovery variable to `v`.
    pub b: f64,
    /// Reset value of `v` after a spike (mV).
    pub c: f64,
    /// Increment of `u` at each spike.
    pub d: f64,
    /// Step (ms); each call takes two Euler half-steps of `dt / 2`.
    pub dt: f64,
}

impl Izhikevich {
    /// Create a neuron with the given parameters at `v = c` and `u = b * c`. Regular
    /// spiking uses a=0.02, b=0.2, c=-65, d=8, dt=1.0 ([`Self::regular_spiking`]).
    pub fn new(a: f64, b: f64, c: f64, d: f64, dt: f64) -> Self {
        Self {
            v: c,
            u: b * c,
            a,
            b,
            c,
            d,
            dt,
        }
    }

    /// Regular spiking preset.
    pub fn regular_spiking() -> Self {
        Self::new(0.02, 0.2, -65.0, 8.0, 1.0)
    }

    /// Advance one step of two Euler half-steps, each updating `v` and `u` from the same
    /// state. Returns 1 when `v` then reaches 30, setting `v` to `c` and adding `d` to
    /// `u`; otherwise 0. The input is not validated.
    pub fn step(&mut self, current: f64) -> i32 {
        let half = self.dt * 0.5;
        for _ in 0..2 {
            let dv = (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + current) * half;
            let du = (self.a * (self.b * self.v - self.u)) * half;
            self.v += dv;
            self.u += du;
        }

        if self.v >= 30.0 {
            self.v = self.c;
            self.u += self.d;
            1
        } else {
            0
        }
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.v = self.c;
        self.u = self.b * self.c;
    }
}

#[cfg(test)]
#[path = "izhikevich_tests.rs"]
mod tests;
