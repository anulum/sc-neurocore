// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety kernel for the Ibarz-Tanaka 2007 map

/// Four-branch Rulkov map from Ibarz et al. (2007), Eqs. 2-3.
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
        let v_next = if self.v < lower {
            -(self.alpha * self.alpha) / 4.0 - self.alpha + current + self.u
        } else if self.v <= 0.0 {
            self.alpha * self.v + (self.v + 1.0) * (self.v + 1.0) + current + self.u
        } else if self.v < upper {
            upper
        } else {
            -1.0
        };
        let u_next = self.u - self.mu * (self.v + 1.0 - self.sigma);
        (v_next.is_finite() && u_next.is_finite()).then_some((
            v_next,
            u_next,
            i32::from(self.v >= upper),
        ))
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
        let v_next = if state.v < lower {
            -(state.alpha * state.alpha) / 4.0 - state.alpha + current + state.u
        } else if state.v <= 0.0 {
            state.alpha * state.v + (state.v + 1.0) * (state.v + 1.0) + current + state.u
        } else if state.v < upper {
            upper
        } else {
            -1.0
        };
        (
            v_next,
            state.u - state.mu * (state.v + 1.0 - state.sigma),
            i32::from(state.v >= upper),
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
    fn source_protocol_golden_counts_are_stable() {
        for (current, expected) in [(0.0, 9), (0.2, 33), (1.0, 195)] {
            let mut state = IbarzTanakaMapNeuron::new();
            let events: i32 = (0..1000).map(|_| state.step(current)).sum();
            assert_eq!(events, expected, "I={current}");
        }
    }
}
