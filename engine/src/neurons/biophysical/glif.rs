// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Teeter 2018 GLIF5 source model

//! Five-state GLIF5 dynamics from Teeter et al. (2018), equations 1–8.

/// Teeter GLIF5 membrane, threshold, after-spike-current, and reset state.
#[derive(Clone, Debug)]
pub struct GLIFNeuron {
    pub v: f64,
    pub theta_spike: f64,
    pub i_asc1: f64,
    pub i_asc2: f64,
    pub theta_voltage: f64,
    pub refractory_remaining: f64,
    pub e_l: f64,
    pub capacitance: f64,
    pub resistance: f64,
    pub theta_inf: f64,
    pub b_spike: f64,
    pub b_voltage: f64,
    pub a_voltage: f64,
    pub k_asc1: f64,
    pub k_asc2: f64,
    pub f_v: f64,
    pub delta_v: f64,
    pub delta_theta_spike: f64,
    pub f_asc1: f64,
    pub f_asc2: f64,
    pub delta_i_asc1: f64,
    pub delta_i_asc2: f64,
    pub refractory_period: f64,
    pub dt: f64,
}

impl GLIFNeuron {
    /// Construct the source-consistent normalized operating profile.
    pub fn new() -> Self {
        Self {
            v: -70.0,
            theta_spike: 0.0,
            i_asc1: 0.0,
            i_asc2: 0.0,
            theta_voltage: 0.0,
            refractory_remaining: 0.0,
            e_l: -70.0,
            capacitance: 10.0,
            resistance: 1.0,
            theta_inf: -50.0,
            b_spike: 0.01,
            b_voltage: 0.01,
            a_voltage: 0.0001,
            k_asc1: 0.1,
            k_asc2: 0.005,
            f_v: 0.0,
            delta_v: 0.0,
            delta_theta_spike: 2.0,
            f_asc1: 1.0,
            f_asc2: 1.0,
            delta_i_asc1: 1.0,
            delta_i_asc2: 0.5,
            refractory_period: 2.0,
            dt: 1.0,
        }
    }

    fn finite(values: &[f64]) -> bool {
        values.iter().all(|value| value.is_finite())
    }

    fn valid(&self) -> bool {
        Self::finite(&[
            self.v,
            self.theta_spike,
            self.i_asc1,
            self.i_asc2,
            self.theta_voltage,
            self.refractory_remaining,
            self.e_l,
            self.capacitance,
            self.resistance,
            self.theta_inf,
            self.b_spike,
            self.b_voltage,
            self.a_voltage,
            self.k_asc1,
            self.k_asc2,
            self.f_v,
            self.delta_v,
            self.delta_theta_spike,
            self.f_asc1,
            self.f_asc2,
            self.delta_i_asc1,
            self.delta_i_asc2,
            self.refractory_period,
            self.dt,
        ]) && self.capacitance > 0.0
            && self.resistance > 0.0
            && self.b_spike > 0.0
            && self.b_voltage > 0.0
            && self.k_asc1 > 0.0
            && self.k_asc2 > 0.0
            && self.dt > 0.0
            && self.refractory_remaining >= 0.0
            && self.refractory_period >= 0.0
    }

    fn decay(rate: f64, dt: f64) -> f64 {
        (-rate * dt).exp()
    }

    fn exponential_convolution(decay_rate: f64, forcing_rate: f64, dt: f64) -> f64 {
        let difference = decay_rate - forcing_rate;
        let scale = 1.0_f64.max(decay_rate.abs()).max(forcing_rate.abs());
        if difference.abs() <= 1e-12 * scale {
            dt * (-decay_rate * dt).exp()
        } else {
            ((-forcing_rate * dt).exp() - (-decay_rate * dt).exp()) / difference
        }
    }

    fn candidate(&self, current: f64) -> Option<(Self, i32)> {
        if !self.valid() || !current.is_finite() {
            return None;
        }
        if self.refractory_remaining > 0.0 {
            let mut next = self.clone();
            next.refractory_remaining = (self.refractory_remaining - self.dt).max(0.0);
            return Some((next, 0));
        }

        let total_current = current + self.i_asc1 + self.i_asc2;
        let membrane_rate = 1.0 / (self.resistance * self.capacitance);
        let membrane_decay = Self::decay(membrane_rate, self.dt);
        let equilibrium_offset = self.resistance * total_current;
        let voltage_offset = self.v - self.e_l;
        let next_offset =
            equilibrium_offset + (voltage_offset - equilibrium_offset) * membrane_decay;
        let next_v = self.e_l + next_offset;
        let next_theta_spike = self.theta_spike * Self::decay(self.b_spike, self.dt);
        let next_i_asc1 = self.i_asc1 * Self::decay(self.k_asc1, self.dt);
        let next_i_asc2 = self.i_asc2 * Self::decay(self.k_asc2, self.dt);
        let threshold_forcing = equilibrium_offset * (1.0 - Self::decay(self.b_voltage, self.dt))
            / self.b_voltage
            + (voltage_offset - equilibrium_offset)
                * Self::exponential_convolution(self.b_voltage, membrane_rate, self.dt);
        let next_theta_voltage = self.theta_voltage * Self::decay(self.b_voltage, self.dt)
            + self.a_voltage * threshold_forcing;
        let mut next = Self {
            v: next_v,
            theta_spike: next_theta_spike,
            i_asc1: next_i_asc1,
            i_asc2: next_i_asc2,
            theta_voltage: next_theta_voltage,
            refractory_remaining: 0.0,
            ..self.clone()
        };
        if !next.valid() {
            return None;
        }
        if next.v <= self.theta_inf + next.theta_spike + next.theta_voltage {
            return Some((next, 0));
        }
        next.v = self.e_l + self.f_v * (next.v - self.e_l) - self.delta_v;
        next.theta_spike += self.delta_theta_spike;
        next.i_asc1 = self.f_asc1 * next.i_asc1 + self.delta_i_asc1;
        next.i_asc2 = self.f_asc2 * next.i_asc2 + self.delta_i_asc2;
        next.refractory_remaining = self.refractory_period;
        next.valid().then_some((next, 1))
    }

    /// Checked source update; invalid input leaves state unchanged.
    pub fn try_step(&mut self, current: f64) -> Option<i32> {
        let (candidate, event) = self.candidate(current)?;
        *self = candidate;
        Some(event)
    }

    /// Network-runner-compatible update with fail-closed invalid-input behavior.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Run a failure-atomic constant-current batch.
    pub fn try_simulate(&mut self, n_steps: usize, current: f64) -> Option<(Vec<f64>, i64)> {
        let mut candidate = self.clone();
        let mut trace = Vec::with_capacity(n_steps);
        let mut events = 0_i64;
        for _ in 0..n_steps {
            events += i64::from(candidate.try_step(current)?);
            trace.push(candidate.v);
        }
        *self = candidate;
        Some((trace, events))
    }

    /// Restore the normalized source-profile state.
    pub fn reset(&mut self) {
        self.v = self.e_l;
        self.theta_spike = 0.0;
        self.i_asc1 = 0.0;
        self.i_asc2 = 0.0;
        self.theta_voltage = 0.0;
        self.refractory_remaining = 0.0;
    }
}

impl Default for GLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_profile_has_source_five_state_contract() {
        let neuron = GLIFNeuron::new();
        assert!(neuron.valid());
        assert_eq!(
            neuron.theta_inf + neuron.theta_spike + neuron.theta_voltage,
            -50.0
        );
    }

    #[test]
    fn reset_rules_and_refractory_state_are_explicit() {
        let mut neuron = GLIFNeuron::new();
        neuron.v = -50.0;
        assert_eq!(neuron.try_step(40.0), Some(1));
        assert_eq!(neuron.v, -70.0);
        assert_eq!(neuron.theta_spike, 2.0);
        assert_eq!(neuron.i_asc1, 1.0);
        assert_eq!(neuron.i_asc2, 0.5);
        assert_eq!(neuron.refractory_remaining, 2.0);
    }

    #[test]
    fn invalid_batch_is_failure_atomic() {
        let mut neuron = GLIFNeuron::new();
        let before = neuron.clone();
        assert!(neuron.try_simulate(4, f64::NAN).is_none());
        assert_eq!(neuron.v, before.v);
        assert_eq!(neuron.theta_spike, before.theta_spike);
    }

    #[test]
    fn source_profile_regimes_are_pinned() {
        for (current, expected) in [(0.0, 0), (22.0, 22), (30.0, 49), (50.0, 80)] {
            let mut neuron = GLIFNeuron::new();
            let (_, events) = neuron.try_simulate(1000, current).expect("valid profile");
            assert_eq!(events, expected, "current={current}");
        }
    }
}
