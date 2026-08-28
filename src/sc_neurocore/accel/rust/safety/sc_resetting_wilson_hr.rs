// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained resetting Wilson-HR safety mirror

//! Independent safety mirror of the retained unit-capacitance project recurrence.

/// Historical SC-NeuroCore resetting polynomial state.
#[derive(Clone, Debug)]
pub struct SCResettingWilsonHRNeuron {
    /// Membrane state.
    pub v: f64,
    /// Recovery state.
    pub r: f64,
    /// Recovery time constant in milliseconds.
    pub tau_r: f64,
    /// Hard-reset event level.
    pub v_peak: f64,
    /// RK4 step in milliseconds.
    pub dt: f64,
}

impl SCResettingWilsonHRNeuron {
    /// Construct the historical project-default state and parameters.
    pub fn new() -> Self {
        Self {
            v: -0.7,
            r: 0.1,
            tau_r: 1.9,
            v_peak: 0.4,
            dt: 0.05,
        }
    }

    fn derivatives(&self, v: f64, r: f64, current: f64) -> Option<(f64, f64)> {
        if !(v.is_finite() && r.is_finite() && current.is_finite()) {
            return None;
        }
        let polynomial = -(17.81 + 47.71 * v + 32.63 * v * v) * (v - 0.55);
        let recovery_current = -26.0 * r * (v + 0.92);
        let candidate = (
            polynomial + recovery_current + current,
            (-r + 1.35 * v + 1.03) / self.tau_r,
        );
        if polynomial.is_finite()
            && recovery_current.is_finite()
            && candidate.0.is_finite()
            && candidate.1.is_finite()
        {
            Some(candidate)
        } else {
            None
        }
    }

    fn rk4_candidate(&self, current: f64) -> Option<(f64, f64)> {
        let v0 = self.v;
        let r0 = self.r;
        let dt = self.dt;
        let k1 = self.derivatives(v0, r0, current)?;
        let k2 = self.derivatives(v0 + 0.5 * dt * k1.0, r0 + 0.5 * dt * k1.1, current)?;
        let k3 = self.derivatives(v0 + 0.5 * dt * k2.0, r0 + 0.5 * dt * k2.1, current)?;
        let k4 = self.derivatives(v0 + dt * k3.0, r0 + dt * k3.1, current)?;
        let candidate = (
            v0 + dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0,
            r0 + dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0,
        );
        if candidate.0.is_finite() && candidate.1.is_finite() {
            Some(candidate)
        } else {
            None
        }
    }

    /// Advance one failure-atomic update and historical hard reset.
    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !validate_sc_resetting_wilson_hr(self) {
            return Err("invalid SC resetting Wilson-HR runtime state");
        }
        if !current.is_finite() {
            return Err("invalid SC resetting Wilson-HR current");
        }
        let (next_v, next_r) = self
            .rk4_candidate(current)
            .ok_or("invalid SC resetting Wilson-HR candidate")?;
        let event = next_v >= self.v_peak;
        self.v = if event { -0.7 } else { next_v };
        self.r = next_r;
        Ok(i32::from(event))
    }

    /// Run one failure-atomic constant-current batch.
    pub fn simulate(
        &mut self,
        n_steps: usize,
        current: f64,
    ) -> Result<(Vec<f64>, i64), &'static str> {
        let mut candidate = self.clone();
        let mut trace = Vec::with_capacity(n_steps);
        let mut events = 0_i64;
        for _ in 0..n_steps {
            events += i64::from(candidate.step(current)?);
            trace.push(candidate.v);
        }
        *self = candidate;
        Ok((trace, events))
    }

    /// Restore the historical dynamic state while retaining configuration.
    pub fn reset(&mut self) {
        self.v = -0.7;
        self.r = 0.1;
    }
}

impl Default for SCResettingWilsonHRNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Return whether state and configuration are finite and admissible.
pub fn validate_sc_resetting_wilson_hr(state: &SCResettingWilsonHRNeuron) -> bool {
    state.v.is_finite()
        && state.r.is_finite()
        && state.tau_r.is_finite()
        && state.tau_r > 0.0
        && state.v_peak.is_finite()
        && state.dt.is_finite()
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn historical_anchor_is_preserved() {
        let mut neuron = SCResettingWilsonHRNeuron::new();
        assert_eq!(neuron.step(2.0).unwrap(), 0);
        assert_eq!(neuron.v, -0.5988676025214146);
        assert_eq!(neuron.r, 0.10134793845659071);
    }

    #[test]
    fn batch_failure_preserves_state() {
        let mut neuron = SCResettingWilsonHRNeuron {
            v: 1.0e103,
            ..Default::default()
        };
        let before = (neuron.v, neuron.r);
        assert!(neuron.simulate(2, 2.0).is_err());
        assert_eq!((neuron.v, neuron.r), before);
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        let mut neuron = SCResettingWilsonHRNeuron::new();
        neuron.dt = 0.0;
        assert!(neuron.step(2.0).is_err());
    }
}
