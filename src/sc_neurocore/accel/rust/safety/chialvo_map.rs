// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Checked Rust safety kernel for the Chialvo map

/// State and parameters for Chialvo's two-dimensional discrete map.
#[derive(Debug, Clone)]
pub struct ChialvoMapNeuron {
    pub x: f64,
    pub y: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub k: f64,
    /// Maintained upward-crossing event level, not a source parameter.
    pub x_threshold: f64,
}

impl ChialvoMapNeuron {
    /// Construct the source-paper parameter set used by the Python reference.
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            a: 0.89,
            b: 0.6,
            c: 0.28,
            k: 0.04,
            x_threshold: 1.0,
        }
    }

    /// Advance one simultaneous map iteration under an additive perturbation.
    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !validate_chialvo_map(self) {
            return Err("invalid Chialvo map runtime state");
        }
        if !current.is_finite() {
            return Err("invalid Chialvo map current");
        }

        let x_previous = self.x;
        let x_squared = self.x * self.x;
        let exponential = safe_exp(self.y - self.x);
        let x_next = x_squared * exponential + self.k + current;
        let y_next = self.a * self.y - self.b * self.x + self.c;
        if !x_next.is_finite() || !y_next.is_finite() {
            return Err("invalid Chialvo map candidate state");
        }

        self.x = x_next;
        self.y = y_next;
        Ok(i32::from(
            x_previous < self.x_threshold && self.x >= self.x_threshold,
        ))
    }

    /// Run checked iterations, leaving the final state in this instance.
    pub fn simulate(
        &mut self,
        n_steps: usize,
        current: f64,
    ) -> Result<(Vec<f64>, i64), &'static str> {
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes = 0_i64;
        for _ in 0..n_steps {
            spikes += i64::from(self.step(current)?);
            trace.push(self.x);
        }
        Ok((trace, spikes))
    }

    /// Restore only state variables; preserve configured parameters.
    pub fn reset(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
    }
}

impl Default for ChialvoMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Return whether all state and parameter fields are finite.
pub fn validate_chialvo_map(state: &ChialvoMapNeuron) -> bool {
    state.x.is_finite()
        && state.y.is_finite()
        && state.a.is_finite()
        && state.b.is_finite()
        && state.c.is_finite()
        && state.k.is_finite()
        && state.x_threshold.is_finite()
}

fn safe_exp(value: f64) -> f64 {
    value.clamp(-500.0, 500.0).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        assert!(validate_chialvo_map(&ChialvoMapNeuron::default()));
    }

    #[test]
    fn matches_independent_source_step() {
        let mut state = ChialvoMapNeuron {
            x: 0.2,
            y: 0.7,
            ..Default::default()
        };
        let expected_x = 0.2_f64 * 0.2_f64 * (0.7_f64 - 0.2_f64).exp() + 0.04 + 0.01;
        let expected_y = 0.89 * 0.7 - 0.6 * 0.2 + 0.28;
        assert_eq!(state.step(0.01), Ok(0));
        assert_eq!(state.x, expected_x);
        assert_eq!(state.y, expected_y);
    }

    #[test]
    fn matches_python_golden_event_counts() {
        for (current, expected) in [(-0.05, 0_i64), (0.0, 26), (0.01, 30), (0.1, 0), (1.0, 1)] {
            let mut state = ChialvoMapNeuron::default();
            let (_trace, spikes) = state.simulate(1000, current).expect("finite source regime");
            assert_eq!(spikes, expected, "current={current}");
        }
    }

    #[test]
    fn rejects_invalid_input_and_candidate_without_mutation() {
        let mut state = ChialvoMapNeuron::default();
        let initial = (state.x, state.y);
        assert!(state.step(f64::NAN).is_err());
        assert_eq!((state.x, state.y), initial);

        state.x = 1.0e308;
        let extreme = (state.x, state.y);
        assert!(state.step(0.0).is_err());
        assert_eq!((state.x, state.y), extreme);
    }

    #[test]
    fn reset_preserves_parameters() {
        let mut state = ChialvoMapNeuron {
            x: 2.0,
            y: -1.0,
            a: 0.8,
            b: 0.4,
            c: 0.2,
            k: 0.03,
            x_threshold: 0.75,
        };
        state.reset();
        assert_eq!((state.x, state.y), (0.0, 0.0));
        assert_eq!(
            (state.a, state.b, state.c, state.k, state.x_threshold),
            (0.8, 0.4, 0.2, 0.03, 0.75)
        );
    }
}
