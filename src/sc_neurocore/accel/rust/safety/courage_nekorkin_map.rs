// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Safety mirror for the Courbage source map

#[derive(Debug, Clone)]
pub struct CourageNekorkinMapNeuron {
    pub x: f64,
    pub y: f64,
    pub m0: f64,
    pub m1: f64,
    pub a: f64,
    pub d: f64,
    pub j: f64,
    pub beta: f64,
    pub eps: f64,
    pub x_threshold: f64,
}

impl CourageNekorkinMapNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            m0: 0.4,
            m1: 0.65,
            a: 0.2,
            d: 0.3,
            j: 0.13,
            beta: 0.25,
            eps: 0.002,
            x_threshold: 0.3,
        }
    }

    fn breakpoints(&self) -> (f64, f64) {
        let am1 = self.a * self.m1;
        let denominator = self.m0 + self.m1;
        (am1 / denominator, (self.m0 + am1) / denominator)
    }

    pub fn f(&self, x: f64) -> f64 {
        let (j_min, j_max) = self.breakpoints();
        if x <= j_min {
            -self.m0 * x
        } else if x < j_max {
            self.m1 * (x - self.a)
        } else {
            -self.m0 * (x - 1.0)
        }
    }

    /// Return -1 without mutation on any invalid input or candidate.
    pub fn step(&mut self, current: f64) -> i32 {
        if !validate_courage_nekorkin_map(self) || !current.is_finite() {
            return -1;
        }
        let x_previous = self.x;
        let field = self.f(self.x);
        let heaviside = if self.x >= self.d { 1.0 } else { 0.0 };
        let x_new = self.x + field - self.y - self.beta * heaviside + current;
        let y_new = self.y + self.eps * (self.x - self.j);
        if !x_new.is_finite() || !y_new.is_finite() {
            return -1;
        }
        let event = i32::from(x_new >= self.x_threshold && x_previous < self.x_threshold);
        self.x = x_new;
        self.y = y_new;
        event
    }

    pub fn reset(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
    }
}

impl Default for CourageNekorkinMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_courage_nekorkin_map(state: &CourageNekorkinMapNeuron) -> bool {
    [
        state.x,
        state.y,
        state.m0,
        state.m1,
        state.a,
        state.d,
        state.j,
        state.beta,
        state.eps,
        state.x_threshold,
    ]
    .iter()
    .all(|value| value.is_finite())
        && state.m0 > 0.0
        && state.m0 < 1.0
        && state.m1 > 0.0
        && state.a > 0.0
        && state.a < 1.0
        && state.d > 0.0
        && state.beta > 0.0
        && state.eps > 0.0
        && state.j > 0.0
        && state.j < state.d
        && {
            let (j_min, j_max) = state.breakpoints();
            j_min < state.d && state.d < j_max
        }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn figure_four_profile_is_valid() {
        let mut neuron = CourageNekorkinMapNeuron::new();
        let events: i32 = (0..1_000).map(|_| neuron.step(0.0)).sum();
        assert_eq!(events, 21);
        assert!(validate_courage_nekorkin_map(&neuron));
    }

    #[test]
    fn invalid_inputs_are_atomic() {
        let mut neuron = CourageNekorkinMapNeuron::new();
        let before = (neuron.x, neuron.y);
        assert_eq!(neuron.step(f64::NAN), -1);
        assert_eq!((neuron.x, neuron.y), before);
        neuron.m0 = 0.0;
        assert_eq!(neuron.step(0.0), -1);
        assert_eq!((neuron.x, neuron.y), before);
    }
}
