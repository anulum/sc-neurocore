// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety kernel for the Ibarz-Tanaka 2007 map

/// Shilnikov-Rulkov (2004) four-branch map profiled by Ibarz et al. (2007).
#[derive(Debug, Clone)]
pub struct IbarzTanakaMapNeuron {
    pub v: f64,
    pub u: f64,
    pub alpha: f64,
    pub mu: f64,
    pub sigma: f64,
}

impl IbarzTanakaMapNeuron {
    pub fn new() -> Self {
        Self {
            v: -1.0,
            u: -0.1,
            alpha: 1.0,
            mu: 0.001,
            sigma: 0.1,
        }
    }

    fn candidate(&self, current: f64) -> Option<(f64, f64, i32)> {
        let lower = -1.0 - self.alpha / 2.0;
        let upper = 1.0 + current + self.u;
        let (v_next, event) = if self.v < lower {
            (
                -(self.alpha * self.alpha) / 4.0 - self.alpha + current + self.u,
                0,
            )
        } else if self.v <= 0.0 {
            (
                self.alpha * self.v + (self.v + 1.0) * (self.v + 1.0) + current + self.u,
                0,
            )
        } else if self.v < upper {
            (upper, 0)
        } else {
            (-1.0, 1)
        };
        let u_next = self.u - self.mu * (self.v + 1.0 - self.sigma);
        (v_next.is_finite() && u_next.is_finite()).then_some((v_next, u_next, event))
    }

    /// Evaluate one checked source step and preserve state on failure.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !validate_ibarz_tanaka_map(self) || !current.is_finite() {
            return Err("invalid Ibarz-Tanaka runtime state");
        }
        let (v_next, u_next, event) = self
            .candidate(current)
            .ok_or("invalid Ibarz-Tanaka candidate")?;
        self.v = v_next;
        self.u = u_next;
        Ok(event)
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Execute a complete checked trace; any rejected step preserves state.
    pub fn simulate_checked(
        &mut self,
        n_steps: usize,
        current: f64,
    ) -> Result<(Vec<f64>, Vec<i32>), &'static str> {
        let mut candidate = self.clone();
        let mut trace = Vec::with_capacity(n_steps);
        let mut event_trace = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            let event = candidate.try_step(current)?;
            trace.push(candidate.v);
            event_trace.push(event);
        }
        self.v = candidate.v;
        self.u = candidate.u;
        Ok((trace, event_trace))
    }

    pub fn reset(&mut self) {
        self.v = -1.0;
        self.u = -0.1;
    }
}

impl Default for IbarzTanakaMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_ibarz_tanaka_map(state: &IbarzTanakaMapNeuron) -> bool {
    state.v.is_finite()
        && state.u.is_finite()
        && state.alpha.is_finite()
        && state.mu.is_finite()
        && state.sigma.is_finite()
        && state.alpha > 0.0
        && state.mu > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_step(state: &IbarzTanakaMapNeuron, current: f64) -> (f64, f64, i32) {
        let lower = -1.0 - state.alpha / 2.0;
        let upper = 1.0 + current + state.u;
        let (v_next, event) = if state.v < lower {
            (
                -(state.alpha * state.alpha) / 4.0 - state.alpha + current + state.u,
                0,
            )
        } else if state.v <= 0.0 {
            (
                state.alpha * state.v + (state.v + 1.0) * (state.v + 1.0) + current + state.u,
                0,
            )
        } else if state.v < upper {
            (upper, 0)
        } else {
            (-1.0, 1)
        };
        (
            v_next,
            state.u - state.mu * (state.v + 1.0 - state.sigma),
            event,
        )
    }

    #[test]
    fn defaults_match_the_source_example() {
        let state = IbarzTanakaMapNeuron::new();
        assert_eq!(
            (state.v, state.u, state.alpha, state.mu, state.sigma),
            (-1.0, -0.1, 1.0, 0.001, 0.1)
        );
        assert!(validate_ibarz_tanaka_map(&state));
    }

    #[test]
    fn all_four_branches_match_an_independent_recurrence() {
        for v in [-2.0, -1.0, 0.5, 1.5] {
            let mut state = IbarzTanakaMapNeuron {
                v,
                u: -0.1,
                ..IbarzTanakaMapNeuron::new()
            };
            let expected = reference_step(&state, 0.2);
            assert_eq!(state.try_step(0.2), Ok(expected.2));
            assert_eq!((state.v, state.u), (expected.0, expected.1));
        }
    }

    #[test]
    fn invalid_input_preserves_state() {
        let mut state = IbarzTanakaMapNeuron::new();
        let before = (state.v, state.u);
        assert!(state.try_step(f64::NAN).is_err());
        assert_eq!((state.v, state.u), before);
    }

    #[test]
    fn earlier_branch_precedence_prevents_false_reset_events() {
        let mut state = IbarzTanakaMapNeuron::new();
        assert!(state.v >= 1.0 - 5.0 + state.u);
        assert_eq!(state.try_step(-5.0), Ok(0));
        assert_ne!(state.v, -1.0);
    }

    #[test]
    fn source_protocol_golden_counts_are_stable() {
        for (current, expected) in [(0.0, 9), (0.2, 33), (1.0, 195)] {
            let mut state = IbarzTanakaMapNeuron::new();
            let events: i32 = (0..1000).map(|_| state.step(current)).sum();
            assert_eq!(events, expected, "I={current}");
        }
    }

    #[test]
    fn checked_batch_returns_complete_events_and_is_failure_atomic() {
        let mut state = IbarzTanakaMapNeuron::new();
        let (trace, event_trace) = state.simulate_checked(1_000, 0.2).unwrap();
        assert_eq!(trace.len(), 1_000);
        assert_eq!(event_trace.len(), 1_000);
        assert_eq!(event_trace.iter().sum::<i32>(), 33);
        assert_eq!(trace.last().copied(), Some(state.v));

        let before = (state.v, state.u);
        assert!(state.simulate_checked(8, f64::INFINITY).is_err());
        assert_eq!((state.v, state.u), before);
    }
}
