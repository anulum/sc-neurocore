// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Naud-Gerstner perfect integrator + preserved SC profile

/// Profile-explicit perfect-integrator state used by the production engine.
#[derive(Clone, Debug)]
pub struct PerfectIntegratorNeuron {
    pub v: f64,
    pub c_m: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub dt: f64,
    /// `true` selects Naud-Gerstner's strict `>` boundary; `false` preserves SC `>=`.
    pub source_profile: bool,
}

/// Complete failure-atomic perfect-integrator batch result.
pub type PerfectIntegratorCompleteTrace = (Vec<f64>, Vec<u8>, f64);

impl PerfectIntegratorNeuron {
    /// Construct the preserved inclusive-threshold SC profile.
    pub fn new(c_m: f64, v_threshold: f64, dt: f64) -> Self {
        Self {
            v: 0.0,
            c_m,
            v_threshold,
            v_reset: 0.0,
            dt,
            source_profile: false,
        }
    }

    /// Construct the normalized Naud-Gerstner 2012 source profile.
    pub fn naud_gerstner_2012() -> Self {
        let mut state = Self::new(1.0, 1.0, 0.1);
        state.source_profile = true;
        state
    }

    /// Report whether configuration and dynamic state obey the selected profile.
    pub fn valid(&self) -> bool {
        self.v.is_finite()
            && self.c_m.is_finite()
            && self.c_m > 0.0
            && self.v_threshold.is_finite()
            && self.v_reset.is_finite()
            && self.v_threshold > self.v_reset
            && if self.source_profile {
                self.v <= self.v_threshold
            } else {
                self.v < self.v_threshold
            }
            && self.dt.is_finite()
            && self.dt > 0.0
    }

    /// Advance one exact held-current step without conflating errors with silence.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() || !self.valid() {
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

    /// Compatibility dispatch for NetworkRunner's uniform non-throwing trait.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Execute a failure-atomic complete batch against a cloned candidate.
    pub fn simulate_complete(
        &self,
        n_steps: usize,
        current: f64,
    ) -> Result<PerfectIntegratorCompleteTrace, &'static str> {
        if !current.is_finite() || !self.valid() {
            return Err("invalid perfect-integrator batch contract");
        }
        let mut candidate = self.clone();
        let mut voltage = Vec::with_capacity(n_steps);
        let mut events = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            let event = candidate.try_step(current)?;
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
        Self::new(1.0, 1.0, 0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::PerfectIntegratorNeuron;

    #[test]
    fn source_and_sc_boundaries_are_distinct() {
        let mut source = PerfectIntegratorNeuron::naud_gerstner_2012();
        let mut sc = PerfectIntegratorNeuron::default();
        assert_eq!(source.try_step(5.0), Ok(0));
        assert_eq!(sc.try_step(5.0), Ok(0));
        assert_eq!(source.try_step(5.0), Ok(0));
        assert_eq!(source.v, source.v_threshold);
        assert_eq!(sc.try_step(5.0), Ok(1));
        assert_eq!(source.try_step(5.0), Ok(1));
    }

    #[test]
    fn source_batch_matches_exact_equality_sequence() {
        let source = PerfectIntegratorNeuron::naud_gerstner_2012();
        let (trace, events, final_v) = source.simulate_complete(6, 5.0).unwrap();
        assert_eq!(trace, vec![0.5, 1.0, 0.0, 0.5, 1.0, 0.0]);
        assert_eq!(events, vec![0, 0, 1, 0, 0, 1]);
        assert_eq!(final_v, 0.0);
    }

    #[test]
    fn invalid_state_and_batch_do_not_mutate() {
        let mut state = PerfectIntegratorNeuron {
            v: 0.25,
            ..PerfectIntegratorNeuron::default()
        };
        assert!(state.try_step(f64::NAN).is_err());
        assert_eq!(state.v, 0.25);
        state.c_m = f64::MIN_POSITIVE;
        state.v_threshold = f64::MAX;
        assert!(state.simulate_complete(2, f64::MAX).is_err());
        assert_eq!(state.v, 0.25);
    }

    #[test]
    fn reset_preserves_parameters_and_profile() {
        let mut state = PerfectIntegratorNeuron::naud_gerstner_2012();
        state.v = 0.5;
        state.reset();
        assert_eq!(state.v, state.v_reset);
        assert!(state.source_profile);
    }
}
