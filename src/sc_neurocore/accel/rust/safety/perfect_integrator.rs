// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Profile-explicit Rust safety for perfect_integrator

#[derive(Debug, Clone)]
pub struct PerfectIntegratorNeuron {
    pub v: f64,
    pub c_m: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub dt: f64,
    pub source_profile: bool,
}

pub type PerfectIntegratorCompleteTrace = (Vec<f64>, Vec<u8>, f64);

impl PerfectIntegratorNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0,
            c_m: 1.0,
            v_threshold: 1.0,
            v_reset: 0.0,
            dt: 0.1,
            source_profile: false,
        }
    }

    pub fn naud_gerstner_2012() -> Self {
        let mut state = Self::new();
        state.source_profile = true;
        state
    }

    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() || !validate_perfect_integrator(self) {
            return Err("perfect-integrator state/current violates its finite profile contract");
        }
        let increment = current * self.dt / self.c_m;
        let candidate = self.v + increment;
        if !increment.is_finite() || !candidate.is_finite() {
            return Err("perfect-integrator voltage candidate became non-finite");
        }
        let crossed = if self.source_profile {
            candidate > self.v_threshold
        } else {
            candidate >= self.v_threshold
        };
        self.v = if crossed { self.v_reset } else { candidate };
        Ok(i32::from(crossed))
    }

    pub fn simulate_complete(
        &self,
        n_steps: usize,
        current: f64,
    ) -> Result<PerfectIntegratorCompleteTrace, &'static str> {
        if !current.is_finite() || !validate_perfect_integrator(self) {
            return Err("invalid perfect-integrator batch contract");
        }
        let mut candidate = self.clone();
        let mut voltage = Vec::with_capacity(n_steps);
        let mut events = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            let event = candidate.step(current)?;
            voltage.push(candidate.v);
            events.push(event as u8);
        }
        Ok((voltage, events, candidate.v))
    }

    pub fn reset(&mut self) {
        self.v = self.v_reset;
    }
}

impl Default for PerfectIntegratorNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_perfect_integrator(state: &PerfectIntegratorNeuron) -> bool {
    state.v.is_finite()
        && state.c_m.is_finite()
        && state.c_m > 0.0
        && state.v_threshold.is_finite()
        && state.v_reset.is_finite()
        && state.v_threshold > state.v_reset
        && if state.source_profile {
            state.v <= state.v_threshold
        } else {
            state.v < state.v_threshold
        }
        && state.dt.is_finite()
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::{validate_perfect_integrator, PerfectIntegratorNeuron};

    #[test]
    fn source_and_sc_exact_thresholds_diverge_as_documented() {
        let mut source = PerfectIntegratorNeuron::naud_gerstner_2012();
        let mut sc = PerfectIntegratorNeuron::new();
        assert_eq!(source.step(5.0), Ok(0));
        assert_eq!(sc.step(5.0), Ok(0));
        assert_eq!(source.step(5.0), Ok(0));
        assert!(validate_perfect_integrator(&source));
        assert_eq!(sc.step(5.0), Ok(1));
        assert_eq!(source.step(5.0), Ok(1));
    }

    #[test]
    fn source_complete_packet_is_aligned() {
        let source = PerfectIntegratorNeuron::naud_gerstner_2012();
        let (trace, events, final_v) = source.simulate_complete(6, 5.0).unwrap();
        assert_eq!(trace, vec![0.5, 1.0, 0.0, 0.5, 1.0, 0.0]);
        assert_eq!(events, vec![0, 0, 1, 0, 0, 1]);
        assert_eq!(final_v, 0.0);
    }

    #[test]
    fn invalid_update_and_batch_are_atomic() {
        let mut state = PerfectIntegratorNeuron::new();
        state.v = 0.25;
        assert!(state.step(f64::NAN).is_err());
        assert_eq!(state.v, 0.25);
        state.c_m = f64::MIN_POSITIVE;
        state.v_threshold = f64::MAX;
        assert!(state.simulate_complete(2, f64::MAX).is_err());
        assert_eq!(state.v, 0.25);
    }

    #[test]
    fn reset_preserves_profile_and_parameters() {
        let mut state = PerfectIntegratorNeuron::naud_gerstner_2012();
        state.v = 0.5;
        state.reset();
        assert_eq!(state.v, state.v_reset);
        assert!(state.source_profile);
    }
}
