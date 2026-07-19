// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Chialvo discrete map neuron

//! Chialvo discrete map neuron.

/// Chialvo 1995 — 2D discrete map neuron.
#[derive(Clone, Debug)]
pub struct ChialvoMapNeuron {
    pub x: f64,
    pub y: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub k: f64,
    pub x_threshold: f64,
}

impl ChialvoMapNeuron {
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
    fn is_valid(&self) -> bool {
        self.x.is_finite()
            && self.y.is_finite()
            && self.a.is_finite()
            && self.b.is_finite()
            && self.c.is_finite()
            && self.k.is_finite()
            && self.x_threshold.is_finite()
    }

    fn safe_exp(value: f64) -> f64 {
        value.clamp(-500.0, 500.0).exp()
    }

    /// Checked Chialvo update used by the production batch dispatcher.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !self.is_valid() {
            return Err("invalid Chialvo map runtime state");
        }
        if !current.is_finite() {
            return Err("invalid Chialvo map current");
        }

        let x_prev = self.x;
        let x_squared = self.x * self.x;
        let exponential = Self::safe_exp(self.y - self.x);
        let x_new = x_squared * exponential + self.k + current;
        let y_new = self.a * self.y - self.b * self.x + self.c;
        if !x_new.is_finite() || !y_new.is_finite() {
            return Err("invalid Chialvo map candidate state");
        }
        self.x = x_new;
        self.y = y_new;
        Ok(if x_prev < self.x_threshold && self.x >= self.x_threshold {
            1
        } else {
            0
        })
    }

    /// Legacy infallible engine-class update. Invalid input leaves the state
    /// unchanged and emits no event; the checked batch API reports the error.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Run checked map iterations, returning the fast-state trace and events.
    pub fn simulate(
        &mut self,
        n_steps: usize,
        current: f64,
    ) -> Result<(Vec<f64>, i64), &'static str> {
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes = 0_i64;
        for _ in 0..n_steps {
            spikes += i64::from(self.try_step(current)?);
            trace.push(self.x);
        }
        Ok((trace, spikes))
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chialvo_matches_independent_source_step() {
        let mut neuron = ChialvoMapNeuron {
            x: 0.2,
            y: 0.7,
            ..Default::default()
        };
        let x = neuron.x;
        let y = neuron.y;
        let expected_x = x * x * (y - x).exp() + neuron.k + 0.01;
        let expected_y = neuron.a * y - neuron.b * x + neuron.c;
        assert_eq!(neuron.try_step(0.01), Ok(0));
        assert_eq!(neuron.x, expected_x);
        assert_eq!(neuron.y, expected_y);
    }

    #[test]
    fn chialvo_matches_python_golden_event_counts() {
        for (current, expected) in [(-0.05, 0_i64), (0.0, 26), (0.01, 30), (0.1, 0), (1.0, 1)] {
            let mut neuron = ChialvoMapNeuron::new();
            let (_trace, spikes) = neuron
                .simulate(1000, current)
                .expect("finite source regime");
            assert_eq!(spikes, expected, "current={current}");
        }
    }

    #[test]
    fn chialvo_rejects_non_finite_input_without_mutation() {
        let mut neuron = ChialvoMapNeuron::new();
        let initial = (neuron.x, neuron.y);
        assert!(neuron.try_step(f64::NAN).is_err());
        assert_eq!((neuron.x, neuron.y), initial);

        neuron.y = f64::INFINITY;
        assert!(neuron.try_step(0.0).is_err());
    }

    #[test]
    fn chialvo_reset_preserves_parameters() {
        let mut neuron = ChialvoMapNeuron {
            x: 2.0,
            y: -1.0,
            a: 0.8,
            b: 0.4,
            c: 0.2,
            k: 0.03,
            x_threshold: 0.75,
        };
        neuron.reset();
        assert_eq!((neuron.x, neuron.y), (0.0, 0.0));
        assert_eq!(
            (neuron.a, neuron.b, neuron.c, neuron.k, neuron.x_threshold),
            (0.8, 0.4, 0.2, 0.03, 0.75)
        );
    }
}
