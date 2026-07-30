// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for Fardet-Levina eLIF

//! Coupled RK4 implementation of the authors' Brian eLIF profile.

const V_MIN: f64 = -200.0;
const V_MAX: f64 = 100.0;
const ENERGY_MAX: f64 = 5.0;

/// Complete Fardet-Levina eLIF state and configuration.
#[derive(Debug, Clone)]
pub struct EnergyLIFNeuron {
    pub v: f64,
    pub epsilon: f64,
    pub capacitance: f64,
    pub g_leak: f64,
    pub e_0: f64,
    pub e_u: f64,
    pub e_d: f64,
    pub e_f: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub alpha: f64,
    pub epsilon_0: f64,
    pub epsilon_c: f64,
    pub delta: f64,
    pub tau_e: f64,
    pub dt: f64,
}

impl Default for EnergyLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl EnergyLIFNeuron {
    /// Construct the pinned author-Brian parameter and initial-state profile.
    pub fn new() -> Self {
        Self {
            v: -61.0,
            epsilon: 0.32,
            capacitance: 100.0,
            g_leak: 9.0,
            e_0: -62.5,
            e_u: -58.5,
            e_d: -40.0,
            e_f: -62.0,
            v_threshold: -59.0,
            v_reset: -62.0,
            alpha: 1.0,
            epsilon_0: 0.5,
            epsilon_c: 0.18,
            delta: 0.01,
            tau_e: 200.0,
            dt: 0.1,
        }
    }

    fn rhs(&self, v: f64, epsilon: f64, current: f64) -> (f64, f64) {
        let leak = self.e_0 + (self.e_u - self.e_0) * (1.0 - epsilon / self.epsilon_0);
        let dv = (self.g_leak * (leak - v) + current) / self.capacitance;
        let production = (1.0 - epsilon / (self.alpha * self.epsilon_0)).powi(3);
        let voltage_cost = (v - self.e_f) / (self.e_d - self.e_f);
        (dv, (production - voltage_cost) / self.tau_e)
    }

    fn candidate(&self, current: f64) -> (f64, f64) {
        let dt = self.dt;
        let k1 = self.rhs(self.v, self.epsilon, current);
        let k2 = self.rhs(
            self.v + 0.5 * dt * k1.0,
            self.epsilon + 0.5 * dt * k1.1,
            current,
        );
        let k3 = self.rhs(
            self.v + 0.5 * dt * k2.0,
            self.epsilon + 0.5 * dt * k2.1,
            current,
        );
        let k4 = self.rhs(self.v + dt * k3.0, self.epsilon + dt * k3.1, current);
        let scale = dt / 6.0;
        (
            self.v + scale * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0),
            self.epsilon + scale * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1),
        )
    }

    /// Advance atomically, returning `-1` for invalid input or state.
    pub fn step(&mut self, current: f64) -> i32 {
        if !self.valid() || !current.is_finite() {
            return -1;
        }
        let (v, epsilon) = self.candidate(current);
        if !(v.is_finite()
            && (V_MIN..=V_MAX).contains(&v)
            && epsilon.is_finite()
            && (0.0..=ENERGY_MAX).contains(&epsilon))
        {
            return -1;
        }
        if v > self.v_threshold && epsilon > self.epsilon_c {
            let after = epsilon - self.delta;
            if !(0.0..=ENERGY_MAX).contains(&after) {
                return -1;
            }
            self.v = self.v_reset;
            self.epsilon = after;
            return 1;
        }
        self.v = v;
        self.epsilon = epsilon;
        0
    }

    /// Return whether the complete state satisfies the source envelope.
    pub fn valid(&self) -> bool {
        [
            self.v,
            self.e_0,
            self.e_u,
            self.e_d,
            self.e_f,
            self.v_threshold,
            self.v_reset,
        ]
        .into_iter()
        .all(f64::is_finite)
            && (V_MIN..=V_MAX).contains(&self.v)
            && (V_MIN..=V_MAX).contains(&self.v_reset)
            && self.epsilon.is_finite()
            && (0.0..=ENERGY_MAX).contains(&self.epsilon)
            && [self.epsilon_0, self.epsilon_c, self.delta]
                .into_iter()
                .all(|x| x.is_finite() && x >= 0.0)
            && [
                self.capacitance,
                self.g_leak,
                self.alpha,
                self.tau_e,
                self.dt,
            ]
            .into_iter()
            .all(|x| x.is_finite() && x > 0.0)
            && self.e_d != self.e_f
            && self.v_threshold > self.v_reset
            && self.dt <= 1.0
            && self.dt <= self.tau_e
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_transition_is_atomic() {
        let mut state = EnergyLIFNeuron::new();
        assert_eq!(state.step(80.0), 0);
        let before = state.clone();
        assert_eq!(state.step(f64::NAN), -1);
        assert_eq!((state.v, state.epsilon), (before.v, before.epsilon));
    }
}
