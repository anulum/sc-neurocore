// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Wilson-Cowan population-rate model

//! Wilson-Cowan excitatory/inhibitory population dynamics.

#![deny(missing_docs)]

/// Wilson-Cowan 1972 — excitatory/inhibitory population rate model.
#[derive(Clone, Debug)]
pub struct WilsonCowanUnit {
    /// Continuous excitatory population rate; it relaxes towards the shifted sigmoid, whose values lie in (-beta, 1 - beta) with beta = 1 / (1 + exp(a \* theta)) (normalised excitatory population activity).
    pub e: f64,
    /// Continuous inhibitory population rate; it relaxes towards the shifted sigmoid, whose values lie in (-beta, 1 - beta) with beta = 1 / (1 + exp(a \* theta)) (normalised inhibitory population activity).
    pub i: f64,
    /// Recurrent excitatory-to-excitatory connection weight (dimensionless).
    pub w_ee: f64,
    /// Inhibitory-to-excitatory connection weight (dimensionless).
    pub w_ei: f64,
    /// Excitatory-to-inhibitory connection weight (dimensionless).
    pub w_ie: f64,
    /// Recurrent inhibitory-to-inhibitory connection weight (dimensionless).
    pub w_ii: f64,
    /// Excitatory population time constant (model time).
    pub tau_e: f64,
    /// Inhibitory population time constant (model time).
    pub tau_i: f64,
    /// Positive steepness of the shifted sigmoid (1/input).
    pub a: f64,
    /// Finite midpoint of the shifted sigmoid (input).
    pub theta: f64,
    /// RK4 step (model time); each call takes one step.
    pub dt: f64,
}

impl WilsonCowanUnit {
    /// Create a unit at `e` = 0.1 and `i` = 0.05 with the default weights, time constants
    /// and sigmoid, and `dt` = 0.1.
    pub fn new() -> Self {
        Self {
            e: 0.1,
            i: 0.05,
            w_ee: 10.0,
            w_ei: 6.0,
            w_ie: 10.0,
            w_ii: 1.0,
            tau_e: 1.0,
            tau_i: 2.0,
            a: 1.2,
            theta: 4.0,
            dt: 0.1,
        }
    }
    fn logistic(&self, z: f64) -> f64 {
        if z >= 0.0 {
            1.0 / (1.0 + (-z).exp())
        } else {
            let exp_z = z.exp();
            exp_z / (1.0 + exp_z)
        }
    }
    fn sigmoid(&self, x: f64) -> f64 {
        self.logistic(self.a * (x - self.theta)) - self.logistic(-self.a * self.theta)
    }
    fn derivatives(&self, e: f64, i: f64, ext_input: f64) -> (f64, f64) {
        let se = self.sigmoid(self.w_ee * e - self.w_ei * i + ext_input);
        let si = self.sigmoid(self.w_ie * e - self.w_ii * i);
        ((-e + se) / self.tau_e, (-i + si) / self.tau_i)
    }
    /// Advance one classical RK4 step with `ext_input`, the external input to the
    /// excitatory population, and return the new `e`.
    ///
    /// The sigmoid is the logistic `1 / (1 + exp(-a * (x - theta)))` shifted down by its
    /// value at 0, so that it is 0 at zero input. Inputs are not validated.
    pub fn step(&mut self, ext_input: f64) -> f64 {
        let e = self.e;
        let i = self.i;
        let (k1_e, k1_i) = self.derivatives(e, i, ext_input);
        let (k2_e, k2_i) = self.derivatives(
            e + 0.5 * self.dt * k1_e,
            i + 0.5 * self.dt * k1_i,
            ext_input,
        );
        let (k3_e, k3_i) = self.derivatives(
            e + 0.5 * self.dt * k2_e,
            i + 0.5 * self.dt * k2_i,
            ext_input,
        );
        let (k4_e, k4_i) = self.derivatives(e + self.dt * k3_e, i + self.dt * k3_i, ext_input);
        self.e = e + self.dt * (k1_e + 2.0 * k2_e + 2.0 * k3_e + k4_e) / 6.0;
        self.i = i + self.dt * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i) / 6.0;
        self.e
    }
    /// Return `e` and `i` to 0.1 and 0.05; the parameters are kept.
    pub fn reset(&mut self) {
        self.e = 0.1;
        self.i = 0.05;
    }
}

impl Default for WilsonCowanUnit {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn population_state_evolves() {
        let mut n = WilsonCowanUnit::new();
        let mut last = n.e;
        let mut changes = 0;
        for _ in 0..500 {
            let e = n.step(5.0);
            if (e - last).abs() > 0.001 {
                changes += 1;
            }
            last = e;
        }
        assert!(changes > 0);
    }

    #[test]
    fn reset_restores_initial_population_state() {
        let mut n = WilsonCowanUnit::new();
        for _ in 0..200 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.e - 0.1).abs() < 1e-10);
        assert!((n.i - 0.05).abs() < 1e-10);
    }

    #[test]
    fn population_remains_finite_under_large_input() {
        let mut n = WilsonCowanUnit::new();
        for _ in 0..5000 {
            n.step(1e3);
        }
        assert!(n.e.is_finite());
        assert!(n.i.is_finite());
    }

    #[test]
    fn nan_input_does_not_panic() {
        WilsonCowanUnit::new().step(f64::NAN);
    }
}
